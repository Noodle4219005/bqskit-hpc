"""This module implements the LEAPSynthesisPass."""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Any
from typing import NamedTuple

import numpy as np
from scipy.stats import linregress

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.location import CircuitLocation
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
from bqskit.runtime.future import RuntimeFuture
from bqskit.runtime.task import PRIORITY_CRITICAL
from bqskit.runtime.task import PRIORITY_SPECULATIVE
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number

_logger = logging.getLogger(__name__)

# ==================== HPC: ordered runahead ========================================
_MIN_PREFIX_SIZE_OVERRIDE = os.environ.get('BQSKIT_MIN_PREFIX_SIZE')
_SPEC_YIELD_ENV = os.environ.get('BQSKIT_SPEC_YIELD')
_OCCUPANCY_STALE_AFTER = 1.0
_SPEC_VALUE_FLOOR = float(os.environ.get('BQSKIT_SPEC_VALUE_FLOOR', '0.02'))
_SPEC_VALUE_WARMUP = int(os.environ.get('BQSKIT_SPEC_VALUE_WARMUP', '32'))
_TASK_QOS = os.environ.get('BQSKIT_TASK_QOS', '1') != '0'
_SPEC_PRIORITY = PRIORITY_SPECULATIVE if _TASK_QOS else PRIORITY_CRITICAL
_SPEC_YIELD = (
    (_SPEC_YIELD_ENV != '0') if _SPEC_YIELD_ENV is not None else not _TASK_QOS
)
# ===================================================================================

# ==================== HPC: K congestion control ====================================
# Grow speculation while the frontier supplies work, then stop probing when it
# is exhausted; a new best result re-arms the controller.
_SPEC_CONTROL = os.environ.get('BQSKIT_SPEC_CONTROL', '1') != '0'
_SPEC_FILL_GROW = float(os.environ.get('BQSKIT_SPEC_FILL_GROW', '0.40'))
_SPEC_FILL_HOLD = float(os.environ.get('BQSKIT_SPEC_FILL_HOLD', '0.20'))
# ===================================================================================

# ==================== HPC: overshoot ===============================================
# Size automatic speculation above free capacity when requested. Auto follows
# contention, while max removes this capacity ceiling.
_SPEC_OVERSHOOT_TEXT = os.environ.get('BQSKIT_SPEC_OVERSHOOT', '4.0')
_SPEC_UNBOUNDED = _SPEC_OVERSHOOT_TEXT.strip().lower() == 'max'
_SPEC_OVERSHOOT_AUTO = _SPEC_OVERSHOOT_TEXT.strip().lower() == 'auto'
_SPEC_OVERSHOOT = (
    1.0 if (_SPEC_UNBOUNDED or _SPEC_OVERSHOOT_AUTO)
    else float(_SPEC_OVERSHOOT_TEXT)
)
_SPEC_OS_FUSE = float(os.environ.get('BQSKIT_SPEC_OS_FUSE', '64.0'))

if _SPEC_OVERSHOOT < 1.0:
    raise ValueError(
        'BQSKIT_SPEC_OVERSHOOT must be >= 1.0, "auto" or "max".',
    )
# ===================================================================================

# ==================== HPC: speculation memo ========================================
_CircuitStructureKey = tuple[
    tuple[Gate, CircuitLocation, tuple[float, ...]],
    ...,
]


class _SpeculationFlight(NamedTuple):
    """A dispatched speculative batch."""

    future: Any
    keys: list[Any]
    batches: list[list[Circuit]]
    owners: list[tuple[int, int]] | None
    epoch: int
    bound_generation: int | None
    n_tasks: int


class _SpeculationMemoEntry(NamedTuple):
    """A deferred expansion and its logical context."""

    epoch: int
    bound_generation: int | None
    successors: list[Circuit]
    results: list[Circuit]
    used: bool
# ===================================================================================

# ==================== HPC: parallel multistart =====================================
def _instantiate_single_start(
    circuit: Circuit,
    target: Any,
    seed: int,
    **kwargs: Any,
) -> Circuit:
    """Instantiate one explicitly seeded starting point."""
    return circuit.instantiate(target, seed=seed, multistarts=1, **kwargs)


class LEAPSynthesisPass(SynthesisPass):
    """
    A pass implementing the LEAP search synthesis algorithm.

    References:
        Ethan Smith, Marc G. Davis, Jeffrey M. Larson, Ed Younis,
        Lindsay Bassman, Wim Lavrijsen, and Costin Iancu. 2022. LEAP:
        Scaling Numerical Optimization Based Synthesis Using an
        Incremental Approach. ACM Transactions on Quantum Computing
        (June 2022). https://doi.org/10.1145/3548693
    """

    def __init__(
        self,
        heuristic_function: HeuristicFunction = AStarHeuristic(),
        layer_generator: LayerGenerator | None = None,
        success_threshold: float = 1e-8,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        max_layer: int | None = None,
        no_progress_layers_allowed: int = 10,
        store_partial_solutions: bool = False,
        partials_per_depth: int = 25,
        min_prefix_size: int = 3,
        async_drain: bool = False,
        parallel_multistart: bool = False,
        instantiate_options: dict[str, Any] = {},
    ) -> None:
        """
        Construct a search-based synthesis pass.

        Args:
            heuristic_function (HeuristicFunction): The heuristic to guide
                search.

            layer_generator (LayerGenerator | None): The successor function
                to guide node expansion. If left as none, then a default
                will be selected before synthesis based on the target
                model's gate set. (Default: None)

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the cost function. (Default: 1e-8)

            cost (CostFunction | None): The cost function that determines
                distance during synthesis. The goal of this synthesis pass
                is to implement circuits for the given unitaries that have
                a cost less than the `success_threshold`.
                (Default: HSDistance())

            max_layer (int): The maximum number of layers to append without
                success before termination. If left as None it will default
                to unlimited. (Default: None)

            no_progress_layers_allowed (int): The maximum number of layers
                allowed without improvement before a warning is issued to
                the user. (Default: 10)

            store_partial_solutions (bool): Whether to store partial solutions
                at different depths inside of the data dict. (Default: False)

            partials_per_depth (int): The maximum number of partials
                to store per search depth. No effect if
                `store_partial_solutions` is False. (Default: 25)

            min_prefix_size (int): The minimum number of layers needed
                to prefix the circuit.

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

        Environment:
            BQSKIT_MAX_ROLLBACKS controls the per-synthesis rollback budget.
            It defaults to BQSKIT_MAX_COMMITTED.
            BQSKIT_EXPAND_K controls ordered speculative expansion. It is
            read when the pass is constructed; unset disables speculation.
            BQSKIT_SPECULATE_MEMO caps the expansion memo and defaults to four
            times BQSKIT_EXPAND_K.
            BQSKIT_DEEPEN_TO sets the ceiling for continuing a bounded search
            after it exhausts its current frontier. The bound doubles up to
            this ceiling, retaining children truncated at the old bound.

        Raises:
            ValueError: If `max_depth` or `min_prefix_size` is nonpositive.
        """
        if _MIN_PREFIX_SIZE_OVERRIDE is not None:
            min_prefix_size = int(_MIN_PREFIX_SIZE_OVERRIDE)

        if not isinstance(heuristic_function, HeuristicFunction):
            raise TypeError(
                'Expected HeursiticFunction, got %s.'
                % type(heuristic_function),
            )

        if layer_generator is not None:
            if not isinstance(layer_generator, LayerGenerator):
                raise TypeError(
                    f'Expected LayerGenerator, got {type(layer_generator)}.',
                )

        if not is_real_number(success_threshold):
            raise TypeError(
                'Expected real number for success_threshold'
                ', got %s' % type(success_threshold),
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

        # ==================== HPC: ordered runahead ================================
        deepen_to = os.environ.get('BQSKIT_DEEPEN_TO')
        self.deepen_to: int | None = None
        if deepen_to is not None:
            try:
                self.deepen_to = int(deepen_to)
            except ValueError as err:
                raise ValueError(
                    'BQSKIT_DEEPEN_TO must be an integer.',
                ) from err
            if self.deepen_to <= 0:
                raise ValueError(
                    'BQSKIT_DEEPEN_TO must be positive, got '
                    f'{self.deepen_to}.',
                )
        # Without max_layer, no child is truncated, so BQSKIT_DEEPEN_TO is
        # inert and does not create an overflow list.

        if min_prefix_size is not None and not is_integer(min_prefix_size):
            raise TypeError(
                'Expected min_prefix_size to be an integer, got %s'
                % type(min_prefix_size),
            )

        if min_prefix_size is not None and min_prefix_size <= 0:
            raise ValueError(
                'Expected min_prefix_size to be positive, got %d.'
                % int(min_prefix_size),
            )

        if not isinstance(instantiate_options, dict):
            raise TypeError(
                'Expected dictionary for instantiate_options, got %s.'
                % type(instantiate_options),
            )

        if not isinstance(async_drain, bool):
            raise TypeError(
                'Expected bool for async_drain, got %s.' % type(async_drain),
            )

        if not isinstance(parallel_multistart, bool):
            raise TypeError(
                'Expected bool for parallel_multistart, got %s.'
                % type(parallel_multistart),
            )

        # K only controls runahead, never frontier order. Auto sizing reserves
        # enough workers for one critical successor batch.

        expand_k_value = os.environ.get('BQSKIT_EXPAND_K')
        expand_k_auto = False
        expand_k = 1
        if expand_k_value is not None:
            if expand_k_value.strip().lower() == 'auto':
                expand_k_auto = True
            else:
                try:
                    expand_k = int(expand_k_value)
                except ValueError as err:
                    raise ValueError(
                        "BQSKIT_EXPAND_K must be an integer or 'auto'.",
                    ) from err
        if expand_k < 1:
            raise ValueError(
                'BQSKIT_EXPAND_K must be positive, got %d.' % expand_k,
            )

        # A worker cannot otherwise see the compiler's worker count.
        try:
            worker_width = int(os.environ.get(
                'BQSKIT_RUNAHEAD_WORKERS', str(os.cpu_count() or 1),
            ))
        except ValueError as err:
            raise ValueError(
                'BQSKIT_RUNAHEAD_WORKERS must be an integer.',
            ) from err
        if worker_width < 1:
            raise ValueError(
                'BQSKIT_RUNAHEAD_WORKERS must be positive, got %d.'
                % worker_width,
            )

        try:
            expand_k_max = int(os.environ.get(
                'BQSKIT_EXPAND_K_MAX', str(max(1, 4 * worker_width)),
            ))
        except ValueError as err:
            raise ValueError(
                'BQSKIT_EXPAND_K_MAX must be an integer.',
            ) from err
        if expand_k_max < 1:
            raise ValueError(
                'BQSKIT_EXPAND_K_MAX must be positive, got %d.'
                % expand_k_max,
            )

        # Eviction only loses work: the logical frontier remains unchanged.
        try:
            speculate_memo = int(os.environ.get(
                'BQSKIT_SPECULATE_MEMO', str(1 << 30),
            ))
        except ValueError as err:
            raise ValueError(
                'BQSKIT_SPECULATE_MEMO must be an integer.',
            ) from err
        if speculate_memo < 1:
            raise ValueError(
                'BQSKIT_SPECULATE_MEMO must be positive, got %d.'
                % speculate_memo,
            )

        self.max_rollbacks = int(os.environ.get(
            'BQSKIT_MAX_ROLLBACKS',
            os.environ.get('BQSKIT_MAX_COMMITTED', '8'),
        ))
        if self.max_rollbacks < 0:
            raise ValueError(
                'Expected BQSKIT_MAX_ROLLBACKS to be nonnegative, got %d.'
                % self.max_rollbacks,
            )
        self.async_drain = async_drain
        self.parallel_multistart = parallel_multistart
        self.expand_k = expand_k
        # How many of this block's own typical improvement gaps to wait
        # before treating it as stalled. Below ~2 a normal gap looks like a
        # stall; far above it the throttle never fires on a real compile.
        try:
            self.stall_patience = float(
                os.environ.get('BQSKIT_STALL_PATIENCE', '3'),
            )
        except ValueError as err:
            raise ValueError(
                'BQSKIT_STALL_PATIENCE must be a number.',
            ) from err
        if self.stall_patience <= 0:
            raise ValueError('BQSKIT_STALL_PATIENCE must be positive.')
        self.expand_k_auto = expand_k_auto
        self.expand_k_max = expand_k_max
        self.worker_width = worker_width
        self.speculate_memo = speculate_memo
        self.heuristic_function = heuristic_function
        self.layer_gen = layer_generator
        self.success_threshold = success_threshold
        self.cost = cost
        self.max_layer = max_layer
        self.no_progress_layers_allowed = no_progress_layers_allowed
        self.min_prefix_size = min_prefix_size
        _fraction = os.environ.get('BQSKIT_MIN_PREFIX_FRACTION')
        if _fraction is None:
            self.min_prefix_fraction: float | None = None
        else:
            try:
                self.min_prefix_fraction = float(_fraction)
            except ValueError as err:
                raise ValueError(
                    'BQSKIT_MIN_PREFIX_FRACTION must be a float, got '
                    f'{_fraction!r}.',
                ) from err
            if not 0.0 < self.min_prefix_fraction <= 1.0:
                raise ValueError(
                    'BQSKIT_MIN_PREFIX_FRACTION must be in (0, 1], got '
                    f'{self.min_prefix_fraction}.',
                )
        # ===========================================================================
        self.instantiate_options: dict[str, Any] = {
            'cost_fn_gen': self.cost,
        }
        self.instantiate_options.update(instantiate_options)
        self.store_partial_solutions = store_partial_solutions
        self.partials_per_depth = partials_per_depth

    async def synthesize(
        self,
        utry: UnitaryMatrix | StateVector | StateSystem,
        data: PassData,
    ) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        iteration = 0

        # Initialize run-dependent options
        instantiate_options = self.instantiate_options.copy()

        # Seed the PRNG
        if 'seed' not in instantiate_options:
            instantiate_options['seed'] = data.seed

        # ==================== HPC: parallel multistart =============================
        def dispatch_batches(
            batches: list[list[Circuit]],
            task_priority: int = PRIORITY_CRITICAL,
        ) -> tuple[RuntimeFuture, list[tuple[int, int]] | None]:
            """Dispatch batches with the requested runtime service class."""
            flat_successors = [
                successor
                for batch in batches
                for successor in batch
            ]

            if (
                not self.async_drain
                and self.parallel_multistart
                and int(instantiate_options.get('multistarts', 1)) > 1
            ):
                num_starts = int(instantiate_options['multistarts'])
                single_options = dict(instantiate_options)
                single_options.pop('multistarts', None)
                single_options.pop('seed', None)
                base_seed = int(instantiate_options.get('seed', 0) or 0)

                flat_circuits = []
                flat_seeds = []
                owner_of_task: list[tuple[int, int]] = []
                for batch_index, batch in enumerate(batches):
                    for successor_index, successor in enumerate(batch):
                        for start_index in range(num_starts):
                            flat_circuits.append(successor)
                            flat_seeds.append(
                                base_seed * num_starts
                                + successor_index * num_starts
                                + start_index,
                            )
                            owner_of_task.append((batch_index, successor_index))

                future = get_runtime().map(
                    _instantiate_single_start,
                    flat_circuits,
                    [utry] * len(flat_circuits),
                    flat_seeds,
                    task_priority=task_priority,
                    cost_hints=[
                        float(4 ** circuit.num_qudits)
                        for circuit in flat_circuits
                    ],
                    **single_options,
                )
                return future, owner_of_task

            future = get_runtime().map(
                Circuit.instantiate,
                flat_successors,
                target=utry,
                task_priority=task_priority,
                cost_hints=[
                    float(4 ** circuit.num_qudits)
                    for circuit in flat_successors
                ],
                **instantiate_options,
            )
            return future, None

        async def collect_batches(
            future: RuntimeFuture,
            batches: list[list[Circuit]],
            owner_of_task: list[tuple[int, int]] | None,
        ) -> list[list[Circuit]]:
            """Collect a dispatched map and restore its batch structure."""
            flat_results = await future
            if owner_of_task is not None:
                best_circuits: list[dict[int, Circuit]] = [
                    {} for _ in batches
                ]
                best_costs: list[dict[int, float]] = [
                    {} for _ in batches
                ]
                for task_index, candidate in enumerate(flat_results):
                    batch_index, successor_index = owner_of_task[task_index]
                    candidate_cost = self.cost.calc_cost(candidate, utry)
                    if (
                        successor_index not in best_costs[batch_index]
                        or candidate_cost
                        < best_costs[batch_index][successor_index]
                    ):
                        best_costs[batch_index][successor_index] = candidate_cost
                        best_circuits[batch_index][successor_index] = candidate

                return [
                    [best_circuits[i][j] for j in range(len(batch))]
                    for i, batch in enumerate(batches)
                ]

            results: list[list[Circuit]] = []
            offset = 0
            for batch in batches:
                end = offset + len(batch)
                results.append(flat_results[offset:end])
                offset = end
            return results

        # ===========================================================================
        # ==================== HPC: speculation memo ================================
        speculation_memo: dict[
            _CircuitStructureKey,
            _SpeculationMemoEntry,
        ] = {}
        speculation_flights: list[_SpeculationFlight] = []
        epoch = 0
        # s: successors per node, measured rather than assumed. None until the
        # first expansion has been seen, which is why round one runs serially.
        succ_ema: float | None = None
        _last_improve_round = 0
        _gap_ema: float | None = None
        _spec_issued = 0
        _spec_hits = 0

        # Contention estimator. `fastest_batch_s` is the uncontended reference:
        # the one round that got the machine to itself. Everything slower than
        # it is other blocks competing for the same workers.
        fastest_batch_s: float | None = None
        contention_ema: float = 1.0

        # ==================== HPC: K congestion control ============================
        # Start above the no-speculation floor so the controller can measure
        # frontier fill before an initial speculative batch is available.
        k_ctl = 2.0
        k_ssthresh = float('inf')
        overshoot = 1.0 if _SPEC_OVERSHOOT_AUTO else _SPEC_OVERSHOOT
        fill_ema: float | None = None
        k_probe_allowed = True
        # ===========================================================================

        def circuit_structure_key(circuit: Circuit) -> _CircuitStructureKey:
            """Return an exact identity for one instantiation input.

            Parameters are part of the key: equal gate layouts can reach different
            local minima, and reusing either result could reorder the frontier.
            """
            return tuple(
                (op.gate, op.location, tuple(op.params)) for op in circuit
            )

        def memoize_speculation(
            structure_key: _CircuitStructureKey,
            dispatch_epoch: int,
            bound_generation: int | None,
            expansion: tuple[list[Circuit], list[Circuit]],
        ) -> None:
            """Insert an expansion, evicting only cached work when necessary."""
            while (
                structure_key not in speculation_memo
                and len(speculation_memo) >= self.speculate_memo
            ):
                oldest_key = next(iter(speculation_memo))
                speculation_memo.pop(oldest_key)
            speculation_memo[structure_key] = _SpeculationMemoEntry(
                dispatch_epoch,
                bound_generation,
                *expansion,
                False,
            )

        def finish_speculation() -> None:
            """Cancel speculative work that outlives this synthesis."""
            nonlocal speculation_flights
            speculation_memo.clear()
            for _flight in speculation_flights:
                # Unfinished speculation must not outlive the synthesis call.
                get_runtime().cancel(_flight.future)
            speculation_flights = []

        # Get layer generator for search
        layer_gen = self._get_layer_gen(data)

        current_max_layer = self.max_layer
        overflow: list[tuple[Circuit, int]] | None = (
            []
            if self.max_layer is not None and self.deepen_to is not None
            else None
        )

        def add_child(circuit: Circuit, layer: int) -> bool:
            """Add a child to the frontier or retain it past the bound."""
            if current_max_layer is None or layer < current_max_layer:
                frontier.add(circuit, layer)
                return True
            if overflow is not None:
                overflow.append((circuit, layer))
            return False

        # ===========================================================================
        # ==================== HPC: initial layer dispatch ==========================
        # Begin the search with an initial layer
        frontier = Frontier(utry, self.heuristic_function)
        initial_layer = layer_gen.gen_initial_layer(utry, data)
        prefetched_initial_successors: list[Circuit] | None = None
        prefetched_initial_future: RuntimeFuture | None = None
        prefetched_initial_owners: list[tuple[int, int]] | None = None
        if self.parallel_multistart and int(
            instantiate_options.get('multistarts', 1),
        ) > 1:
            # Its successors are certain after initial instantiation, so prefetch
            # them without changing the frontier or commit order.
            num_starts = int(instantiate_options['multistarts'])
            single_options = dict(instantiate_options)
            single_options.pop('multistarts', None)
            single_options.pop('seed', None)
            base_seed = int(instantiate_options.get('seed', 0) or 0)

            # gen_successors reads only structure and connectivity, while one
            # random start overwrites every parameter during instantiation.
            # Therefore the templates below are exactly the ones that would be
            # generated after the initial layer converges; retain the future
            # outside the speculation memo and consume it on the first pop.
            prefetched_initial_successors = list(
                layer_gen.gen_successors(initial_layer, data),
            )
            if prefetched_initial_successors:
                flat_circuits = []
                flat_seeds = []
                prefetched_initial_owners = []
                for successor_index, successor in enumerate(
                    prefetched_initial_successors,
                ):
                    for start_index in range(num_starts):
                        flat_circuits.append(successor)
                        flat_seeds.append(
                            base_seed * num_starts
                            + successor_index * num_starts
                            + start_index,
                        )
                        prefetched_initial_owners.append(
                            (0, successor_index),
                        )
                prefetched_initial_future = get_runtime().map(
                    _instantiate_single_start,
                    flat_circuits,
                    [utry] * len(flat_circuits),
                    flat_seeds,
                    cost_hints=[
                        float(4 ** circuit.num_qudits)
                        for circuit in flat_circuits
                    ],
                    **single_options,
                )

            initial_seeds = [
                base_seed * num_starts + start_index
                for start_index in range(num_starts)
            ]
            initial_candidates = await get_runtime().map(
                _instantiate_single_start,
                [initial_layer] * num_starts,
                [utry] * num_starts,
                initial_seeds,
                cost_hints=[
                    float(4 ** initial_layer.num_qudits)
                ] * num_starts,
                **single_options,
            )
            initial_layer = min(
                initial_candidates,
                key=lambda candidate: self.cost.calc_cost(candidate, utry),
            )
        else:
            initial_layer.instantiate(utry, **instantiate_options)
        frontier.add(initial_layer, 0)

        def abandon_prefetched_initial() -> None:
            """Cancel first-round work if no first expansion consumes it."""
            nonlocal prefetched_initial_future
            if prefetched_initial_future is not None:
                # Runtime futures own a mailbox until they are awaited. Dropping
                # the Python reference would leave it owned by this synthesis;
                # cancel removes that mailbox and sends cancellation to every
                # outstanding task, so an early success or empty frontier
                # cannot leave work behind or deadlock a later runtime wait.
                get_runtime().cancel(prefetched_initial_future)
                prefetched_initial_future = None

        # ===========================================================================
        # Track best circuit, initially the initial layer
        best_dist = self.cost.calc_cost(initial_layer, utry)
        best_circ = initial_layer
        best_layer = 0
        best_dists = [best_dist]

        best_layers = [0]
        last_prefix_layer = 0
        n_rollbacks = 0

        # Track partial solutions
        psols: dict[int, list[tuple[Circuit, float]]] = {}

        _logger.debug(f'Search started, initial layer has cost: {best_dist}.')

        # Evalute initial layer
        if best_dist < self.success_threshold:
            _logger.debug('Successful synthesis with 0 layers.')
            abandon_prefetched_initial()
            return initial_layer

        # Record layers that have been warned about
        # to avoid duplicate warnings
        warned_layers: list[int] = []

        # ==================== HPC: ordered runahead ================================
        # Main loop.
        #
        # An emptied frontier used to mean the search was over, and the exit
        # below returns a circuit that is guaranteed to have failed
        # success_threshold. Now it first tries going back: the committed
        # prefixes are still there, and a prefix that led nowhere is exactly
        # the branch worth reconsidering.
        #
        # KNOWN LIMIT: best_layers / best_dists are deliberately NOT rolled
        # back, because they mean "best seen so far". They are also what
        # check_leap_condition regresses on, so after a rollback the plateau
        # heuristic sees history from the abandoned branch. Harmless while
        # rollbacks are rare; revisit if n_rollbacks turns out to be large.
        while True:
            if frontier.empty():
                if (
                    frontier.committed_depth() > 0
                    and n_rollbacks < self.max_rollbacks
                ):
                    # Backjump, not chronological backtracking. The most
                    # recent commit is rarely the one that caused the
                    # failure; the one that was forced hardest against the
                    # frontier's own ordering is. Jump there and let
                    # rollback_to invalidate everything below it.
                    _states = frontier.committed_states()
                    _target = max(
                        range(len(_states)),
                        key=lambda i: (
                            _states[i].get('regret', 0.0)
                            if isinstance(_states[i], dict) else 0.0
                        ),
                    )
                    restored_count, restored_state = frontier.rollback_to(
                        _target,
                    )
                    epoch += 1
                    last_prefix_layer = (
                        restored_state.get('last_prefix_layer', 0)
                        if isinstance(restored_state, dict)
                        else (restored_state or 0)
                    )
                    n_rollbacks += 1
                    _logger.debug(
                        'Frontier emptied; rolling back %d elements '
                        '(rollback %d/%d).',
                        restored_count,
                        n_rollbacks,
                        self.max_rollbacks,
                    )
                    continue
                if (
                    overflow
                    and current_max_layer is not None
                    and self.deepen_to is not None
                    and current_max_layer < self.deepen_to
                ):
                    # Rollback explores a cheaper different branch at this
                    # depth; only deepen after the rollback budget is spent.
                    old_max_layer = current_max_layer
                    current_max_layer = min(
                        current_max_layer * 2,
                        self.deepen_to,
                    )
                    for circuit, layer in overflow:
                        frontier.add(circuit, layer)
                    overflow.clear()
                    continue
                break

            # Read once per round, used twice: to size K below, and to decide
            # whether the stall throttle has anyone to yield to. None means the
            # occupancy broadcast is missing or stale.
            measured_idle: int | None = None

            # ==================== HPC: overshoot ===================================
            # K changes scheduling only; frontier pops remain ordered. Auto
            # derives its lead from contention, while max removes capacity caps.
            s = 1
            if self.expand_k_auto:
                if succ_ema is None:
                    effective_k = 1
                else:
                    s = max(1, round(succ_ema))
                    if _SPEC_OVERSHOOT_AUTO:
                        overshoot = max(
                            1.0,
                            min(contention_ema, _SPEC_OS_FUSE),
                        )
                    measured_idle = self._measured_idle_workers()
                    if measured_idle is not None:
                        share = (
                            float(self.expand_k_max) * float(s)
                            if _SPEC_UNBOUNDED else max(
                                float(s),
                                float(measured_idle) * overshoot,
                            )
                        )
                    else:
                        share = max(
                            float(s),
                            self.worker_width / max(1.0, contention_ema),
                        )

                    # Timely hits, not occupancy, bound useful speculation.
                    if _spec_issued >= _SPEC_VALUE_WARMUP and not _SPEC_UNBOUNDED:
                        p = _spec_hits / _spec_issued
                        if p <= 0.0:
                            value_k = 2.0
                        elif p >= 1.0:
                            value_k = float(self.expand_k_max)
                        else:
                            value_k = 1.0 + math.log(
                                _SPEC_VALUE_FLOOR,
                            ) / math.log(p)
                        share = min(share, max(float(s), value_k * s))
                    effective_k = max(
                        1,
                        min(
                            self.expand_k_max,
                            1 + int((share - s) // s),
                        ),
                    )
            else:
                effective_k = self.expand_k
            # =======================================================================

            # A stalled block yields scarce capacity to still-improving blocks.
            if effective_k > 1 and _gap_ema is not None:
                _since = iteration - _last_improve_round
                _stall = _since / max(1.0, _gap_ema)
                if _stall > self.stall_patience:
                    _scarce = (
                        measured_idle is None
                        or measured_idle < s
                    )
                    if _scarce:
                        effective_k = max(
                            1,
                            int(effective_k / (_stall / self.stall_patience)),
                        )

            # Never await speculation: the critical path must remain ordered.
            if speculation_flights:
                still_flying: list[_SpeculationFlight] = []
                for flight in speculation_flights:
                    if not flight.future._done:
                        still_flying.append(flight)
                        continue
                    flight_results = await collect_batches(
                        flight.future, flight.batches, flight.owners,
                    )
                    for structure_key, node_successors, node_results in zip(
                        flight.keys, flight.batches, flight_results,
                    ):
                        if flight.epoch != epoch:
                            continue
                        memoize_speculation(
                            structure_key,
                            flight.epoch,
                            flight.bound_generation,
                            (node_successors, node_results),
                        )
                speculation_flights = still_flying

            popped = []
            successors = []
            successor_layers = []
            popped_expansions: list[
                tuple[list[Circuit], list[Circuit] | None],
            ] = []
            first_expansion_future: RuntimeFuture | None = None
            first_expansion_owners: list[tuple[int, int]] | None = None

            while not frontier.empty():
                top_circuit, top_layer = frontier.pop()
                popped.append((top_circuit, top_layer))
                # Keep each successor's parent layer with its batch.
                if prefetched_initial_future is not None:
                    # Initial expansion is certain and was prefetched once.
                    first_expansion_future = prefetched_initial_future
                    first_expansion_owners = prefetched_initial_owners
                    prefetched_initial_future = None
                    node_successors = prefetched_initial_successors or []
                    successors.extend(node_successors)
                    successor_layers.extend([top_layer] * len(node_successors))
                    n_node_successors = len(node_successors)
                elif effective_k >= 2:
                    structure_key = circuit_structure_key(top_circuit)
                    memo_entry = speculation_memo.get(structure_key)
                    if memo_entry is not None and memo_entry.epoch != epoch:
                        speculation_memo.pop(structure_key)
                        memo_entry = None

                    in_flight_batch: list[Circuit] | None = None
                    for _flight in speculation_flights:
                        if (
                            _flight.epoch == epoch
                            and structure_key in _flight.keys
                        ):
                            in_flight_batch = _flight.batches[
                                _flight.keys.index(structure_key)
                            ]
                            break
                    if memo_entry is not None:
                        node_successors = memo_entry.successors
                        # A boundary child is logically overflow until the
                        # bound rises; publishing its speculative result now
                        # would expose state the bounded serial search cannot.
                        #
                        # Only that. Deepening does NOT invalidate E(A): the
                        # evaluation does not depend on the bound at all, so
                        # the bound decides *publication eligibility*, never
                        # *evaluation validity*. Requiring the entry's own
                        # bound_generation to be set and still current made a
                        # raise of the bound discard results that remained
                        # perfectly correct -- throwing away numerical work to
                        # economise on a resource that is not scarce. The
                        # field is kept for diagnostics and no longer gates.
                        bound_is_eligible = (
                            current_max_layer is None
                            or top_layer + 1 < current_max_layer
                        )
                        if bound_is_eligible:
                            node_results = memo_entry.results
                            speculation_memo[structure_key] = (
                                memo_entry._replace(used=True)
                            )
                            _spec_hits += 1
                        else:
                            node_results = None
                    elif in_flight_batch is not None:
                        # Never await speculation for a logical pop. Reuse
                        # only its immutable templates and submit fresh
                        # critical instantiations below.
                        node_successors = in_flight_batch
                        node_results = None
                    else:
                        node_successors = list(
                            layer_gen.gen_successors(top_circuit, data),
                        )
                        node_results = None
                    popped_expansions.append((node_successors, node_results))
                    successors.extend(node_successors)
                    successor_layers.extend(
                        [top_layer] * len(node_successors),
                    )
                    n_node_successors = len(node_successors)
                else:
                    n_node_successors = 0
                    _plain = list(
                        layer_gen.gen_successors(top_circuit, data),
                    )
                    for successor in _plain:
                        successors.append(successor)
                        successor_layers.append(top_layer)
                        n_node_successors += 1

                # s, updated on the critical node in BOTH branches. Measuring
                # it only under expand_k >= 2 would deadlock 'auto': round one
                # runs serially, so the estimate would never be taken and K
                # would stay at 1 forever. A node with no successors is a dead
                # end and says nothing about the typical width, so it is not
                # allowed to drag the estimate to zero and inflate K.
                if n_node_successors > 0:
                    succ_ema = (
                        float(n_node_successors) if succ_ema is None
                        else 0.5 * succ_ema + 0.5 * n_node_successors
                    )

                break

            critical_future: RuntimeFuture | None = None
            critical_batches: list[list[Circuit]] = []
            critical_owners: list[tuple[int, int]] | None = None
            if effective_k >= 2 and first_expansion_future is None:
                critical_batches = [
                    node_successors
                    for node_successors, node_results in popped_expansions
                    if node_results is None and node_successors
                ]
                if critical_batches:
                    # QoS normally keeps critical work ahead of speculation.
                    if _SPEC_YIELD and speculation_flights:
                        _cancelled = sum(
                            f.n_tasks for f in speculation_flights
                        )
                        for _flight in speculation_flights:
                            get_runtime().cancel(_flight.future)
                        speculation_flights = []
                    # Queue critical work first so workers cannot choose newly
                    # dispatched speculation ahead of the search path.
                    critical_future, critical_owners = dispatch_batches(
                        critical_batches,
                    )
                # Refill rather than wait for a batch boundary. The budget
                # reserves one critical batch (s tasks) at all times, so the
                # resource rule holds continuously instead of once per batch.
                _s = max(1, round(succ_ema)) if succ_ema else len(successors)
                _s = max(1, _s)
                _in_flight = sum(f.n_tasks for f in speculation_flights)
                # ==================== HPC: K congestion control ====================
                # The controller uses frontier fill rather than a free-core
                # reading, which only describes capacity before this round.
                k_eff = effective_k
                if _SPEC_CONTROL:
                    k_eff = int(max(1.0, min(
                        float(self.expand_k_max), k_ctl,
                    )))
                budget = max(0, (k_eff * _s) - _s - _in_flight)
                # ===================================================================
                acc = 0
                if budget > 0:
                    queued_keys: set[_CircuitStructureKey] = set()
                    for _flight in speculation_flights:
                        queued_keys.update(_flight.keys)
                    next_keys: list[_CircuitStructureKey] = []
                    next_batches: list[list[Circuit]] = []
                    # Look past cached entries so refills add new work.
                    peek = len(speculation_memo) + k_eff * 2 + 8
                    for _, circuit, _ in frontier.peek(peek):
                        if acc >= budget:
                            break
                        structure_key = circuit_structure_key(circuit)
                        memo_entry = speculation_memo.get(structure_key)
                        if (
                            (
                                memo_entry is not None
                                and memo_entry.epoch == epoch
                            )
                            or structure_key in queued_keys
                        ):
                            continue
                        node_successors = list(
                            layer_gen.gen_successors(circuit, data),
                        )
                        if not node_successors:
                            continue
                        if acc + len(node_successors) > budget:
                            break
                        acc += len(node_successors)
                        queued_keys.add(structure_key)
                        next_keys.append(structure_key)
                        next_batches.append(node_successors)

                # Slow-start while the frontier supplies work, then grow
                # linearly. Exhaustion freezes probing until search advances.
                if _SPEC_CONTROL and budget <= 0 and k_probe_allowed:
                    k_ctl = max(
                        2.0,
                        min(k_ctl * 2.0, float(self.expand_k_max)),
                    )
                elif _SPEC_CONTROL and budget > 0:
                    fill = acc / float(budget)
                    fill_ema = (
                        fill if fill_ema is None
                        else 0.7 * fill_ema + 0.3 * fill
                    )
                    if k_probe_allowed:
                        if fill_ema >= _SPEC_FILL_GROW and k_ctl < k_ssthresh:
                            k_ctl = min(
                                k_ctl * 2.0,
                                float(self.expand_k_max),
                            )
                        elif fill_ema >= _SPEC_FILL_HOLD:
                            k_ctl = min(
                                k_ctl + 1.0,
                                float(self.expand_k_max),
                            )
                        else:
                            k_ssthresh = max(2.0, k_ctl / 2.0)
                            k_ctl = k_ssthresh
                            k_probe_allowed = False

                if next_batches:
                        # Count NODES speculated on, matching what a timely
                        # hit is counted against: one node's speculation either
                        # gets used before the critical path reaches it, or it
                        # does not.
                        _spec_issued += len(next_batches)
                        _future, _owners = dispatch_batches(
                            next_batches, _SPEC_PRIORITY,
                        )
                        speculation_flights.append(_SpeculationFlight(
                            _future, next_keys, next_batches, _owners,
                            epoch, current_max_layer, acc,
                        ))
            iteration += 1

            if len(successors) == 0:
                continue

            # Instantiate successors
            if first_expansion_future is not None:
                # The parent was popped only after its initial instantiation
                # completed, but this prefetch has been running since before
                # that wait. `collect_batches` restores successor order and
                # picks the best start exactly like an ordinary fan-out.
                circuits = (await collect_batches(
                    first_expansion_future,
                    [successors],
                    first_expansion_owners,
                ))[0]
            elif effective_k >= 2:
                # The one wait that cannot be hidden: the logical search
                # cannot publish until these land. Speculation is judged by how
                # much of this it removes, not by how busy it keeps the machine.
                _t_stall = time.perf_counter()
                pending_results = (
                    await collect_batches(
                        critical_future,
                        critical_batches,
                        critical_owners,
                    )
                    if critical_future is not None else []
                )
                _batch_s = time.perf_counter() - _t_stall
                # A critical batch is s tasks that run concurrently when the
                # machine is free, so its wall time is ~one instantiate. Longer
                # means the tasks queued behind somebody else.
                if critical_batches and _batch_s > 0:
                    if fastest_batch_s is None or _batch_s < fastest_batch_s:
                        fastest_batch_s = _batch_s
                    _c = max(1.0, _batch_s / fastest_batch_s)
                    # Smoothed: one slow round is noise, a sustained rise is a
                    # neighbour. Falls faster than it rises so the tail is
                    # picked up quickly rather than being throttled by history.
                    _w = 0.5 if _c > contention_ema else 0.25
                    contention_ema = (
                        (1 - _w) * contention_ema + _w * _c
                    )

                circuits = []
                pending_index = 0
                for node_successors, node_results in popped_expansions:
                    if node_results is None:
                        if node_successors:
                            node_results = pending_results[pending_index]
                            pending_index += 1
                        else:
                            node_results = []
                    circuits.extend(node_results)
            elif self.async_drain:
                # Preserve successor order while consuming completed results.
                drain_future = get_runtime().map(
                    Circuit.instantiate,
                    successors,
                    target=utry,
                    cost_hints=[
                        float(4 ** circuit.num_qudits)
                        for circuit in successors
                    ],
                    **instantiate_options,
                )
                circuits = [None] * len(successors)  # type: ignore
                num_outstanding = len(successors)
                while num_outstanding > 0:
                    for index, result in await get_runtime().next(
                        drain_future,
                    ):
                        circuits[index] = result
                        num_outstanding -= 1
            elif self.parallel_multistart and int(
                instantiate_options.get('multistarts', 1),
            ) > 1:
                # Independent starts run as separate tasks; choose each
                # successor's result with the unchanged cost function.
                num_starts = int(instantiate_options['multistarts'])
                single_options = dict(instantiate_options)
                single_options.pop('multistarts', None)
                single_options.pop('seed', None)
                base_seed = int(instantiate_options.get('seed', 0) or 0)

                flat_circuits = []
                flat_seeds = []
                owner_of_task = []
                for successor_index, successor in enumerate(successors):
                    for start_index in range(num_starts):
                        flat_circuits.append(successor)
                        flat_seeds.append(
                            base_seed * num_starts
                            + successor_index * num_starts
                            + start_index,
                        )
                        owner_of_task.append(successor_index)

                flat_results = await get_runtime().map(
                    _instantiate_single_start,
                    flat_circuits,
                    [utry] * len(flat_circuits),
                    flat_seeds,
                    cost_hints=[
                        float(4 ** circuit.num_qudits)
                        for circuit in flat_circuits
                    ],
                    **single_options,
                )

                # Keep the best start per successor, by the same cost the
                # sequential path sorts on.
                circuits = [None] * len(successors)  # type: ignore
                best_costs: list[float | None] = [None] * len(successors)
                for task_index, candidate in enumerate(flat_results):
                    owner = owner_of_task[task_index]
                    candidate_cost = self.cost.calc_cost(candidate, utry)
                    if (
                        best_costs[owner] is None
                        or candidate_cost < best_costs[owner]  # type: ignore
                    ):
                        best_costs[owner] = candidate_cost
                        circuits[owner] = candidate
            else:
                circuits = await get_runtime().map(
                    Circuit.instantiate,
                    successors,
                    target=utry,
                    cost_hints=[
                        float(4 ** circuit.num_qudits)
                        for circuit in successors
                    ],
                    **instantiate_options,
                )

            # Evaluate successors
            for win_index, circuit in enumerate(circuits):
                # Depth of *this* successor's parent, not the round's first.
                layer = successor_layers[win_index]
                dist = self.cost.calc_cost(circuit, utry)

                if dist < self.success_threshold:
                    _logger.debug(
                        f'Successful synthesis with {layer + 1} layers.',
                    )
                    if self.store_partial_solutions:
                        data['psols'] = psols
                    finish_speculation()
                    abandon_prefetched_initial()
                    return circuit

                if self.check_new_best(layer + 1, dist, best_layer, best_dist):
                    _gap = iteration - _last_improve_round
                    _gap_ema = (
                        float(_gap) if _gap_ema is None
                        else 0.6 * _gap_ema + 0.4 * _gap
                    )
                    _last_improve_round = iteration
                    plural = '' if layer == 0 else 's'
                    _logger.debug(
                        f'New best circuit found with {layer + 1} layer{plural}'
                        f' and cost: {dist:.12e}.',
                    )
                    best_dist = dist
                    best_circ = circuit
                    best_layer = layer + 1

                    # Re-probe after search progress refills the frontier.
                    k_probe_allowed = True

                    # ==================== HPC: prefix commit =======================
                    if self.check_leap_condition(
                        layer + 1,
                        best_dist,
                        best_layers,
                        best_dists,
                        last_prefix_layer,
                    ):
                        _logger.debug(f'Prefix formed at {layer + 1} layers.')
                        # Save the pre-prefix state before updating the commit
                        # depth, so rollback returns to the branch point.
                        _remaining = frontier.topk_costs(1)
                        _regret = (
                            frontier.score(circuit) - _remaining[0]
                            if _remaining else 0.0
                        )
                        frontier.commit({
                            'last_prefix_layer': last_prefix_layer,
                            'regret': _regret,
                            'layer': layer + 1,
                        })
                        last_prefix_layer = layer + 1
                        add_child(circuit, layer + 1)
                    # ===============================================================
                if self.store_partial_solutions:
                    if layer not in psols:
                        psols[layer] = []

                    psols[layer].append((circuit.copy(), dist))

                    if len(psols[layer]) > self.partials_per_depth:
                        psols[layer].sort(key=lambda x: x[1])
                        del psols[layer][-1]

                add_child(circuit, layer + 1)
            layer_diff = abs(best_layer - layer)
            if (
                layer_diff % self.no_progress_layers_allowed == 0
                and layer_diff > 0
                and layer not in warned_layers
            ):
                _logger.warning(
                    'No improvement after '
                    f'{self.no_progress_layers_allowed} layers.',
                )
                warned_layers.append(layer)

        # ===========================================================================
        _logger.warning('Frontier emptied.')
        _logger.warning(
            'Returning best known circuit with %d layer%s and cost: %e.'
            % (best_layer, '' if best_layer == 1 else 's', best_dist),
        )
        if self.store_partial_solutions:
            data['psols'] = psols

        finish_speculation()
        abandon_prefetched_initial()
        return best_circ

    def check_new_best(
        self,
        layer: int,
        dist: float,
        best_layer: int,
        best_dist: float,
    ) -> bool:
        """
        Check if the new layer depth and dist are a new best node.

        Args:
            layer (int): The current layer in search.

            dist (float): The current distance in search.

            best_layer (int): The current best layer in the search tree.

            best_dist (float): The current best distance in search.
        """
        better_layer = (
            dist < best_dist
            and (
                best_dist >= self.success_threshold
                or layer <= best_layer
            )
        )
        better_dist_and_layer = (
            dist < self.success_threshold and layer < best_layer
        )
        return better_layer or better_dist_and_layer

    def check_leap_condition(
        self,
        new_layer: int,
        best_dist: float,
        best_layers: list[int],
        best_dists: list[float],
        last_prefix_layer: int,
    ) -> bool:
        """
        Return true if the leap condition is satisfied.

        Args:
            new_layer (int): The current layer in search.

            best_dist (float): The current best distance in search.

            best_layers (list[int]): The list of layers associated
                with recorded best distances.

            best_dists (list[float]): The list of recorded best
                distances.

            last_prefix_layer (int): The last layer a prefix was formed.
        """

        with np.errstate(invalid='ignore', divide='ignore'):
            # Calculate predicted best value
            m, y_int, _, _, _ = linregress(best_layers, best_dists)

        predicted_best = m * (new_layer) + y_int

        # Track new values
        best_layers.append(new_layer)
        best_dists.append(best_dist)

        if np.isnan(predicted_best):
            return False

        # Compute difference between actual value
        delta = predicted_best - best_dist

        _logger.debug(
            'Predicted best value %f for new best best with delta %f.'
            % (predicted_best, delta),
        )

        layers_added = new_layer - last_prefix_layer
        return delta < 0 and layers_added >= self.effective_min_prefix_size

    @property
    def effective_min_prefix_size(self) -> int:
        """
        The prefix threshold actually in force.

        A fraction of the depth bound when both are set, otherwise the
        absolute constant. At least 1, because a threshold of 0 would form a
        prefix on every new best and commit the search to its first guess.
        """
        if self.max_layer is None or self.min_prefix_fraction is None:
            return self.min_prefix_size
        return max(1, int(self.max_layer * self.min_prefix_fraction))

    def _measured_idle_workers(self) -> int | None:
        """Return fresh free-core capacity, or None when unavailable."""
        try:
            cache = get_runtime().get_cache()
        except Exception:
            return None
        entry = cache.get('__bqskit_occupancy__')
        if not entry:
            return None
        try:
            if len(entry) >= 4:
                _idle, _total, free, stamp = entry[:4]
            else:
                free, _total, stamp = entry[:3]
        except (TypeError, ValueError):
            return None
        if time.monotonic() - stamp > _OCCUPANCY_STALE_AFTER:
            return None
        return int(free)

    def _get_layer_gen(self, data: PassData) -> LayerGenerator:
        """
        Set the layer generator.

        If a layer generator has been passed into the constructor, then that
        layer generator will be used. Otherwise, a default layer generator will
        be selected by the gateset.

        If seeds are passed into the data dict, then a SeedLayerGenerator will
        wrap the previously selected layer generator.
        """
        # TODO: Deduplicate this code with qsearch synthesis
        layer_gen = self.layer_gen or data.gate_set.build_mq_layer_generator()

        # Priority given to seeded synthesis
        if 'seed_circuits' in data:
            return SeedLayerGenerator(data['seed_circuits'], layer_gen)

        return layer_gen
