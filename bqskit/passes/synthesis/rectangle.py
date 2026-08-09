"""This module implements the RectangleSynthesisPass.

An anytime, breadth-first replacement for LEAP whose parallel batch width grows
with the search instead of being capped by the coupling graph's degree.

WHY
---
LEAP is an anytime search built on best-first search, and it commits: when its
prefix condition fires it calls ``frontier.clear()``, discarding every other
branch. That prune is not sound -- a discarded branch can contain the only
structure that expresses the target -- so LEAP is neither complete nor
asymptotically optimal, and its ``min_prefix_size`` knob has to be tuned.

It is also what caps parallelism. LEAP pops ONE node per iteration and maps over
that node's successors, and a node's successor count is the coupling graph's
edge count: 3 at block width 3, 6 at width 4, 10 at width 5. No worker pool can
be busier than that, which is why a 113-worker pool was measured delivering
about 1.5x.

Rectangle search (Lemons, Ruml, Holte, and Linares Lopez, AAAI 2024,
arXiv:2312.12554) fixes both at once. It keeps ONE open list per depth level and
each iteration expands one node at every already-explored depth plus `aspect`
new deeper levels, so the explored region is a rectangle that widens and deepens
together. Nothing is discarded; pruning is only against the incumbent.

WHY IT FITS SYNTHESIS PARTICULARLY WELL
---------------------------------------
1. Depth IS the objective. A search node at depth L is a circuit with L
   two-qubit gates, so ``g(n) = L`` exactly. Rectangle search prunes on
   ``f(n) >= g(incumbent)``; taking ``h = 0`` makes that ``L >= L_incumbent``,
   which is sound WITHOUT needing the heuristic to be admissible. The heuristic
   is then used only to order within a level, where soundness does not depend
   on it. Both of the paper's theorems -- completeness, and optimality of the
   last solution returned -- therefore hold here with the inadmissible
   AStarHeuristic still doing the ordering.

2. The paper's own conclusion is that rectangle search suits "problems featuring
   deep local minima". Structure-space local minima are exactly the failure mode
   measured for LEAP/QSearch: a target where only one of the 27 reachable
   3-qubit structures can express it, and no amount of extra numerical restarts
   on the wrong structure helps.

3. The batch. One iteration expands one node at each active level, and those
   expansions are independent, so the whole row is a single ``map()``:

       batch = (levels + deepen_count) * successors_per_node * multistarts

   At depth 10 on a width-5 block with 4 multistarts that is 800 concurrent
   instantiations against LEAP's 10, and it grows every iteration.

RELAXED EXPANSION ORDER
-----------------------
The paper's pseudocode expands level 0, then 1, then 2 ... within an iteration,
so a child produced at level i can be popped at level i+1 in the SAME iteration.
Doing that faithfully would serialise the row and put the batch back to one
node's successors. This pass expands the whole row concurrently instead and
lands the children in the next iteration's lists.

That is safe. Theorem 1 (completeness) rests on starting from the initial layer,
terminating only when every level is empty, and not pruning non-duplicate nodes
before a solution exists. Theorem 2 (the last solution returned is optimal)
rests on pruning only nodes that cannot beat the incumbent, plus the same
termination condition. Neither argument mentions the order in which nodes are
expanded, so relaxing it preserves both -- and unlike an out-of-order async
scheme, one barrier per iteration keeps the search deterministic.

CAVEAT, STATED PLAINLY
----------------------
Completeness here is modulo instantiation: a node is judged infeasible when the
numerical optimiser fails to reach `success_threshold`, which is not the same as
it being infeasible. Duplicate detection is by STRUCTURE (which gates on which
qubits, parameters ignored), on the argument that two instances of one structure
should converge to the same solution. Structurally different circuits realising
the same unitary are not detected as duplicates.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.passes.search.frontier import Frontier
from bqskit.passes.search.generator import LayerGenerator
from bqskit.passes.search.generators.seed import SeedLayerGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.passes.search.heuristics import AStarHeuristic
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.runtime import get_runtime
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number

_logger = logging.getLogger(__name__)

# --- per-dispatch probe -----------------------------------------------------
# Active only when BQPROF_DIR is set (see scripts/bqprof_site/sitecustomize.py
# for the sibling probes this follows the convention of). One JSON line per
# iteration that actually dispatches a batch, appended and flushed
# immediately -- Compiler terminates the runtime server on exit, so an
# atexit-flushed buffer would never be written. A probe failure must never
# break the search, so every write is wrapped in a bare except.
#
# The file is opened LAZILY, on the first record actually emitted, not at
# import time. Every bqskit process that imports this module -- including
# ones that never construct or run RectangleSynthesisPass -- would otherwise
# create an empty rect_<pid>.jsonl merely by importing it.
_BQPROF_DIR = os.environ.get('BQPROF_DIR')
_bqprof_fh = None
_bqprof_open_attempted = False


def _bqprof_emit(record: dict[str, Any]) -> None:
    global _bqprof_fh, _bqprof_open_attempted
    if not _BQPROF_DIR:
        return
    if _bqprof_fh is None:
        if _bqprof_open_attempted:
            return
        _bqprof_open_attempted = True
        try:
            os.makedirs(_BQPROF_DIR, exist_ok=True)
            _bqprof_fh = open(
                os.path.join(_BQPROF_DIR, 'rect_%d.jsonl' % os.getpid()),
                'a', buffering=1,
            )
        except Exception:
            return
    try:
        _bqprof_fh.write(json.dumps(record) + '\n')
    except Exception:
        pass


def structure_key(circuit: Circuit) -> tuple[Any, ...]:
    """A hashable identity for a circuit's STRUCTURE, ignoring parameters.

    Two candidates that place the same gates on the same qudits in the same
    order differ only in continuous parameters, and instantiation drives those
    to the same optimum, so expanding both is wasted worker time. Parameters are
    therefore excluded deliberately -- including them would make every node
    unique and disable duplicate detection entirely.
    """
    return tuple(
        (op.gate.name, tuple(op.location))
        for op in circuit
    )


class RectangleSynthesisPass(SynthesisPass):
    """
    Anytime beam synthesis over per-depth open lists.

    A drop-in alternative to :class:`LEAPSynthesisPass` with the same
    constructor surface, minus `min_prefix_size` (there is no prefix to size)
    and plus `aspect` and `max_batch`.

    References:
        Sofia Lemons, Wheeler Ruml, Robert C. Holte, and Carlos Linares Lopez.
        2024. Rectangle Search: An Anytime Beam Search. In Proceedings of AAAI.
        arXiv:2312.12554
    """

    def __init__(
        self,
        heuristic_function: HeuristicFunction = AStarHeuristic(),
        layer_generator: LayerGenerator | None = None,
        success_threshold: float = 1e-8,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        max_layer: int | None = None,
        aspect: int = 1,
        max_batch: int | None = None,
        store_partial_solutions: bool = False,
        partials_per_depth: int = 25,
        detect_duplicates: bool = True,
        instantiate_options: dict[str, Any] = {},
    ) -> None:
        """
        Construct a rectangle-search synthesis pass.

        Args:
            heuristic_function (HeuristicFunction): Orders nodes WITHIN a depth
                level. It does not participate in pruning, so it need not be
                admissible. (Default: AStarHeuristic())

            layer_generator (LayerGenerator | None): The successor function. If
                None, chosen from the target model's gate set. (Default: None)

            success_threshold (float): Distance below which a candidate counts
                as a solution. (Default: 1e-8)

            cost (CostFunctionGenerator): Distance during synthesis.
                (Default: HilbertSchmidtResidualsGenerator())

            max_layer (int | None): Stop growing beyond this depth. Unlike
                LEAP's, this bound also bounds memory, since every level's list
                is retained. (Default: None)

            aspect (int): New depth levels opened per iteration. 1 gives the
                square schedule of the paper's default. Larger values reach a
                first solution sooner and widen the batch faster, at the cost of
                exploring shallow levels less. (Default: 1)

            max_batch (int | None): Cap on instantiations dispatched in one
                iteration. The point of this pass is a batch that grows, so the
                cap exists only to keep a deep search from dispatching more work
                than the pool can hold. None means uncapped. (Default: None)

            store_partial_solutions (bool): Store per-depth partials in the
                data dict. (Default: False)

            partials_per_depth (int): Partials kept per depth. (Default: 25)

            detect_duplicates (bool): Skip successors whose structure has been
                expanded before. See `structure_key`. (Default: True)

            instantiate_options (dict[str, Any]): Passed to
                circuit.instantiate. (Default: {})

        Raises:
            ValueError: If `max_layer` or `aspect` is nonpositive.
        """
        if not isinstance(heuristic_function, HeuristicFunction):
            raise TypeError(
                'Expected HeursiticFunction, got %s.' % type(heuristic_function),
            )

        if layer_generator is not None:
            if not isinstance(layer_generator, LayerGenerator):
                raise TypeError(
                    f'Expected LayerGenerator, got {type(layer_generator)}.',
                )

        if not is_real_number(success_threshold):
            raise TypeError(
                'Expected real number for success_threshold, got %s'
                % type(success_threshold),
            )

        if not isinstance(cost, CostFunctionGenerator):
            raise TypeError(
                'Expected cost to be a CostFunctionGenerator, got %s'
                % type(cost),
            )

        if max_layer is not None and not is_integer(max_layer):
            raise TypeError(
                'Expected max_layer to be an integer, got %s' % type(max_layer),
            )

        if max_layer is not None and max_layer <= 0:
            raise ValueError(
                'Expected max_layer to be positive, got %d.' % int(max_layer),
            )

        if not is_integer(aspect):
            raise TypeError(
                'Expected aspect to be an integer, got %s' % type(aspect),
            )

        if aspect <= 0:
            raise ValueError(
                'Expected aspect to be positive, got %d.' % int(aspect),
            )

        if not isinstance(instantiate_options, dict):
            raise TypeError(
                'Expected dictionary for instantiate_options, got %s.'
                % type(instantiate_options),
            )

        self.heuristic_function = heuristic_function
        self.layer_gen = layer_generator
        self.success_threshold = success_threshold
        self.cost = cost
        self.max_layer = max_layer
        self.aspect = aspect
        self.max_batch = max_batch
        self.detect_duplicates = detect_duplicates
        self.instantiate_options: dict[str, Any] = {'cost_fn_gen': self.cost}
        self.instantiate_options.update(instantiate_options)
        self.store_partial_solutions = store_partial_solutions
        self.partials_per_depth = partials_per_depth

    async def synthesize(
        self,
        utry: UnitaryMatrix | StateVector | StateSystem,
        data: PassData,
    ) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        instantiate_options = self.instantiate_options.copy()
        if 'seed' not in instantiate_options:
            instantiate_options['seed'] = data.seed

        layer_gen = self._get_layer_gen(data)

        initial_layer = layer_gen.gen_initial_layer(utry, data)
        initial_layer.instantiate(utry, **instantiate_options)

        best_dist = self.cost.calc_cost(initial_layer, utry)
        best_circ = initial_layer
        best_layer = 0
        psols: dict[int, list[tuple[Circuit, float]]] = {}

        if best_dist < self.success_threshold:
            _logger.debug('Successful synthesis with 0 layers.')
            return initial_layer

        # One open list per depth. `openlists[i]` holds candidates with i
        # two-qubit gates; index IS g(n), which is what makes the incumbent
        # prune sound without an admissible heuristic.
        openlists: dict[int, Frontier] = {0: Frontier(utry, self.heuristic_function)}
        openlists[0].add(initial_layer, 0)

        seen: set[tuple[Any, ...]] = set()
        if self.detect_duplicates:
            seen.add(structure_key(initial_layer))

        # The incumbent's layer count is the pruning bound: nothing at or below
        # it can improve on it. None until a solution exists, so that until then
        # nothing is pruned -- which is what Theorem 1 requires.
        incumbent: Circuit | None = None
        incumbent_layer: int | None = None

        deepest = 0        # deepest level opened so far
        deepen = 1         # expansions taken at each newly opened level
        iteration = 0

        while openlists:
            iteration += 1
            n_discarded_by_pop = 0

            # --- select the row -------------------------------------------
            # One node from every already-explored level, then `deepen` from
            # the levels opened this iteration. Popping is cheap and serial;
            # the expensive part is the single dispatch below.
            picks: list[tuple[Circuit, int]] = []

            for level in sorted(openlists):
                node, n_disc = self._pop_useful(openlists, level, incumbent_layer)
                n_discarded_by_pop += n_disc
                if node is not None:
                    picks.append((node, level))

            for _ in range(self.aspect):
                if self.max_layer is not None and deepest + 1 > self.max_layer:
                    break
                deepest += 1
                openlists.setdefault(deepest, Frontier(utry, self.heuristic_function))

            deepen_this_iter = deepen
            for _ in range(deepen):
                node, n_disc = self._pop_useful(openlists, deepest, incumbent_layer)
                n_discarded_by_pop += n_disc
                if node is None:
                    break
                picks.append((node, deepest))

            deepen += self.aspect

            # Drop levels that are exhausted AND can no longer receive children.
            for level in [k for k, f in openlists.items() if f.empty()]:
                if level > deepest or level == 0:
                    openlists.pop(level, None)

            if not picks:
                if all(f.empty() for f in openlists.values()):
                    break
                continue

            # --- expand the row into ONE batch ----------------------------
            batch: list[Circuit] = []
            parents: list[int] = []
            n_dropped_admission = 0
            n_dup_skipped = 0
            for node, level in picks:
                if self.max_layer is not None and level + 1 > self.max_layer:
                    n_dropped_admission += 1
                    continue
                if incumbent_layer is not None and level + 1 >= incumbent_layer:
                    n_dropped_admission += 1
                    continue    # sound: g alone already matches the incumbent
                for succ in layer_gen.gen_successors(node, data):
                    if self.detect_duplicates:
                        key = structure_key(succ)
                        if key in seen:
                            n_dup_skipped += 1
                            continue
                        seen.add(key)
                    batch.append(succ)
                    parents.append(level + 1)
                    if self.max_batch is not None and len(batch) >= self.max_batch:
                        break
                if self.max_batch is not None and len(batch) >= self.max_batch:
                    break

            if not batch:
                if all(f.empty() for f in openlists.values()):
                    break
                continue

            _logger.debug(
                'Rectangle iteration %d: %d levels, batch of %d.'
                % (iteration, len(openlists), len(batch)),
            )

            if _BQPROF_DIR:
                _bqprof_emit({
                    'iteration': iteration,
                    'aspect': self.aspect,
                    'incumbent_layer': incumbent_layer,
                    'deepest': deepest,
                    'deepen': deepen_this_iter,
                    'openlist_sizes': {
                        lvl: len(f) for lvl, f in openlists.items()
                    },
                    'pick_levels': [lvl for _, lvl in picks],
                    'batch_size': len(batch),
                    'n_dropped_admission': n_dropped_admission,
                    'n_dup_skipped': n_dup_skipped,
                    'n_discarded_by_pop': n_discarded_by_pop,
                    't': time.perf_counter(),
                    'pid': os.getpid(),
                })

            circuits = await get_runtime().map(
                Circuit.instantiate,
                batch,
                target=utry,
                **instantiate_options,
            )

            # --- evaluate -------------------------------------------------
            for circuit, layer in zip(circuits, parents):
                dist = self.cost.calc_cost(circuit, utry)

                if dist < self.success_threshold:
                    # Anytime: record and keep going. The prune bound tightens,
                    # so the search shrinks rather than stopping.
                    if incumbent_layer is None or layer < incumbent_layer:
                        incumbent, incumbent_layer = circuit, layer
                        best_circ, best_dist, best_layer = circuit, dist, layer
                        _logger.debug(
                            'New incumbent with %d layers.' % layer,
                        )
                    continue

                if dist < best_dist and incumbent is None:
                    best_dist, best_circ, best_layer = dist, circuit, layer

                if self.store_partial_solutions:
                    psols.setdefault(layer, []).append((circuit.copy(), dist))
                    if len(psols[layer]) > self.partials_per_depth:
                        psols[layer].sort(key=lambda x: x[1])
                        del psols[layer][-1]

                if incumbent_layer is not None and layer + 1 >= incumbent_layer:
                    continue    # its children could not beat the incumbent
                if self.max_layer is not None and layer > self.max_layer:
                    continue

                openlists.setdefault(
                    layer, Frontier(utry, self.heuristic_function),
                ).add(circuit, layer)
                deepest = max(deepest, layer)

        if self.store_partial_solutions:
            data['psols'] = psols

        if incumbent is not None:
            _logger.debug(
                'Rectangle search finished: %d layers after %d iterations.'
                % (incumbent_layer, iteration),
            )
            return incumbent

        _logger.warning(
            'Rectangle search exhausted. Returning best known circuit with '
            '%d layers and cost: %e.' % (best_layer, best_dist),
        )
        return best_circ

    def _pop_useful(
        self,
        openlists: dict[int, Frontier],
        level: int,
        incumbent_layer: int | None,
    ) -> tuple[Circuit | None, int]:
        """Pop the best node at `level`, skipping ones the incumbent dominates.

        With g(n) = level and h = 0 the test is f(n) >= g(incumbent), which is
        the paper's prune specialised to this domain -- sound regardless of what
        the ordering heuristic does.

        Returns `(circuit, n_discarded)`: `n_discarded` counts nodes popped and
        thrown away by the incumbent guard before either a usable node was
        found or the frontier ran dry. Since the guard is a function of
        `level` alone (not of the popped node), once it fires it fires for
        every remaining node at this level -- so this can drain the entire
        level's frontier in one call.
        """
        frontier = openlists.get(level)
        if frontier is None:
            return None, 0
        n_discarded = 0
        while not frontier.empty():
            circuit, _extra = frontier.pop()
            if incumbent_layer is not None and level >= incumbent_layer:
                n_discarded += 1
                continue
            return circuit, n_discarded
        return None, n_discarded

    def _get_layer_gen(self, data: PassData) -> LayerGenerator:
        """Mirror of LEAPSynthesisPass._get_layer_gen so the two are swappable."""
        layer_gen = self.layer_gen or data.gate_set.build_mq_layer_generator()

        if 'seed_circuits' in data:
            return SeedLayerGenerator(data['seed_circuits'], layer_gen)

        return layer_gen
