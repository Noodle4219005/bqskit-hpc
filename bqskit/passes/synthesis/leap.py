"""This module implements the LEAPSynthesisPass."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import numpy as np
from scipy.stats import linregress

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


# Optional measurement for completed instantiation work that follows LEAP's
# first successful cost evaluation in a map batch. The source-level hooks below
# are deliberately guarded so the ordinary LEAP path remains byte-for-byte
# equivalent in its control flow when the probe is disabled.
_LEAPWASTE_DIR = os.environ.get('BQPROF_LEAPWASTE_DIR') or os.environ.get(
    'BQPROF_LEAPWASTE_AGG_DIR',
)
_LEAPWASTE_AGGREGATE = os.environ.get('BQPROF_LEAPWASTE_AGGREGATE') == '1'
_GDFS = os.environ.get('BQPROF_GDFS') == '1'
# Optional benchmark-only override; unset preserves the normal LEAP defaults.
_MIN_PREFIX_SIZE_OVERRIDE = os.environ.get('BQSKIT_MIN_PREFIX_SIZE')

# P0-c: the gate on the ordered-speculative-frontier design (docs/04 5.6.4).
#
# Ordered speculation dispatches the top-K frontier nodes at once but may only
# commit them in the baseline's order. After committing P0 its children go
# back into the frontier, so the next node actually popped may be a child
# rather than P1 -- and then the work spent on P1..P{K-1} is squashed.
#
# The whole approach therefore turns on one number: how many of the next
# several commits come from ONE dispatch. That is measurable on an ordinary
# run with a counter, before any speculation exists. Rank r means a speculator
# of width r+1 had that result ready; "miss" means the node did not exist at
# snapshot time and no width could have covered it.
#
# The snapshot is held for BQPROF_SPEC_WINDOW pops. The first version
# refreshed it every pop, which only asked "was the frontier minimum popped"
# -- true unless a fresh child displaced it -- and gave a curve flat at 87.2%
# from width 1 to 128. That is the depth-1 prefetch rate, worth taking
# in-flight from 1 to 2, and says nothing about whether width K pays.
_SPEC_PROBE_DIR = os.environ.get('BQPROF_SPEC_DIR')
_SPEC_PROBE_K = int(os.environ.get('BQPROF_SPEC_K', '32'))
# How many pops one snapshot must serve. A speculator of width K dispatches
# once and hopes the next several commits all come from that dispatch, so the
# snapshot has to be held across them rather than refreshed each pop.
#
# The window must default to K, not to a constant. A dispatch of width K can
# only be repaid by the next K commits, so a window of 8 observes at most 8
# of the 128 results a K=128 dispatch produced -- and the marginal rank
# histogram then reports a near-perfect hit rate for wide K purely because
# top-128 almost certainly contains the next 8 pops. That curve recommends a
# large K while never having measured what a large K costs.
_SPEC_WINDOW = int(os.environ.get('BQPROF_SPEC_WINDOW', str(_SPEC_PROBE_K)))
_SPEC_STATE: dict[str, Any] = {}


def _spec_probe_record(depth: int, rank: int | None) -> None:
    """Count one pop by its position and rank relative to the snapshot.

    ``depth`` is 0-based: 0 is the first commit after the dispatch, which a
    speculator gets for free because it is the node that triggered it. The
    JOINT distribution of (depth, rank) is what the design question needs --
    coverage of a width-w dispatch is how many of the next w commits held
    rank < w, and no marginal over either axis can reconstruct that.
    """
    if not _SPEC_PROBE_DIR:
        return
    if _SPEC_STATE.get('pid') != os.getpid():
        _SPEC_STATE.clear()
        _SPEC_STATE.update({
            'pid': os.getpid(), 'k': _SPEC_PROBE_K, 'window': _SPEC_WINDOW,
            'n_pops': 0, 'n_hits': 0, 'n_misses': 0,
            'rank_hist': {}, 'depth_rank_hist': {}, 'depth_miss_hist': {},
            'last_flush': 0.0,
        })
    _SPEC_STATE['n_pops'] += 1
    if rank is None:
        _SPEC_STATE['n_misses'] += 1
        misses = _SPEC_STATE['depth_miss_hist']
        dkey = str(depth)
        misses[dkey] = misses.get(dkey, 0) + 1
    else:
        _SPEC_STATE['n_hits'] += 1
        key = str(rank)
        hist = _SPEC_STATE['rank_hist']
        hist[key] = hist.get(key, 0) + 1
        jkey = f'{depth}:{rank}'
        joint = _SPEC_STATE['depth_rank_hist']
        joint[jkey] = joint.get(jkey, 0) + 1

    now = time.monotonic()
    if now - _SPEC_STATE['last_flush'] < 20.0:
        return
    try:
        os.makedirs(_SPEC_PROBE_DIR, exist_ok=True)
        path = os.path.join(_SPEC_PROBE_DIR, f'spec_{os.getpid()}.json')
        tmp = f'{path}.tmp'
        summary = {k: v for k, v in _SPEC_STATE.items() if k != 'last_flush'}
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, separators=(',', ':'))
        os.replace(tmp, path)
        _SPEC_STATE['last_flush'] = now
    except Exception:
        pass
# P0-d: the gate on multi-prefix LEAP.
#
# When check_leap_condition fires, LEAP does frontier.clear() and re-seeds
# with the single new best. That is a destructive commit: every alternative
# continuation is discarded with no way back. The measured trigger rate at
# msz=4 is 0.043%, so it almost never happens there -- but docs/19 showed
# that lowering min_prefix_size makes it happen 13.65% of the time and buys
# 3.5x, short of the >5.3x needed. The suspicion is that the shortfall is
# quality being paid for those discarded alternatives.
#
# Multi-prefix LEAP keeps K of them and searches them in parallel. Whether
# that is worth anything is one number: how close the discarded candidates
# were to the one kept. Close => the choice is near a coin flip and keeping
# several recovers what the aggressive threshold costs. Far => the kept one
# was genuinely better and multi-prefix only burns workers.
#
# Measurable with no implementation, on an ordinary run.
_PREFIX_PROBE_DIR = os.environ.get('BQPROF_PREFIX_DIR')
_PREFIX_PROBE_K = int(os.environ.get('BQPROF_PREFIX_K', '16'))
_PREFIX_FH_STATE: dict[str, Any] = {'pid': None, 'fh': None}


def _prefix_probe_record(
    layer: int,
    frontier: Any,
    kept: Any,
    n_frontier: int,
) -> None:
    """Record the cost of the kept continuation against the discarded ones."""
    if not _PREFIX_PROBE_DIR:
        return
    try:
        discarded = frontier.topk_costs(_PREFIX_PROBE_K)
        if not discarded:
            return
        record = {
            'layer': layer,
            'n_frontier': n_frontier,
            # Scored in the same frontier, so it is directly comparable with
            # the discarded costs rather than being a different quantity
            # that happens to be a float.
            'kept_cost': frontier.score(kept),
            'discarded_costs': [round(c, 9) for c in discarded],
        }
        pid = os.getpid()
        if _PREFIX_FH_STATE['pid'] != pid:
            os.makedirs(_PREFIX_PROBE_DIR, exist_ok=True)
            _PREFIX_FH_STATE['fh'] = open(
                os.path.join(_PREFIX_PROBE_DIR, f'prefix_{pid}.jsonl'),
                'a',
                buffering=1,
            )
            _PREFIX_FH_STATE['pid'] = pid
        _PREFIX_FH_STATE['fh'].write(json.dumps(record) + '\n')
    except Exception:
        pass


_LEAPWASTE_COUNTER = 0
_LEAPWASTE_FH_STATE = {'pid': None, 'fh': None}
_LEAPWASTE_AGG_STATE: dict[str, Any] = {}
_GDFS_HIST_KEYS = (
    'gdfs_descent_hist',
    'gdfs_backtrack_jump_hist',
    'gdfs_post_commit_depth_hist',
    'gdfs_solutions_in_round_hist',
    'gdfs_win_index_hist',
)


def _leapwaste_new_synth_id() -> str:
    """Return a process-qualified identifier for one ``synthesize`` call."""
    global _LEAPWASTE_COUNTER
    _LEAPWASTE_COUNTER += 1
    return f'{os.getpid()}-{_LEAPWASTE_COUNTER}'


def _leapwaste_emit(record: dict[str, Any]) -> None:
    """Append one LEAP iteration record using a fork-safe JSONL handle."""
    if not _LEAPWASTE_DIR or _LEAPWASTE_AGGREGATE:
        return
    try:
        pid = os.getpid()
        if _LEAPWASTE_FH_STATE['pid'] != pid:
            os.makedirs(_LEAPWASTE_DIR, exist_ok=True)
            _LEAPWASTE_FH_STATE['fh'] = open(
                os.path.join(_LEAPWASTE_DIR, f'leapiter_{pid}.jsonl'),
                'a',
                buffering=1,
            )
            _LEAPWASTE_FH_STATE['pid'] = pid
        _LEAPWASTE_FH_STATE['fh'].write(json.dumps(record) + '\n')
    except Exception:
        pass


def _leapwaste_aggregate_flush(force: bool = False) -> None:
    """Atomically refresh one compact aggregate file for this process."""
    if not _LEAPWASTE_DIR or not _LEAPWASTE_AGGREGATE:
        return
    now = time.monotonic()
    if not force and now - _LEAPWASTE_AGG_STATE.get('last_flush', 0.0) < 30.0:
        return
    pid = os.getpid()
    if _LEAPWASTE_AGG_STATE.get('pid') != pid:
        _LEAPWASTE_AGG_STATE.clear()
        _LEAPWASTE_AGG_STATE.update({
            'pid': pid,
            'last_flush': 0.0,
            'iterations': 0,
            'synth_calls': 0,
            'prefix_formed': 0,
            'terminated': 0,
            'sum_n_after_win': 0,
            'n_after_win_count': 0,
            'sum_n_cleared': 0,
            'n_cleared_count': 0,
            'frontier_len_hist': {},
            'n_successors_hist': {},
            'layer_hist': {},
            'max_layer_per_synth_hist': {},
            'active_synth_max_layer': None,
            'sum_n_pruned': 0,
            'rounds_where_prune_fired': 0,
            'sum_popped_per_round': 0,
            'n_rounds': 0,
            'gdfs_descent_hist': {},
            'gdfs_backtrack_jump_hist': {},
            'gdfs_post_commit_depth_hist': {},
            'gdfs_solutions_in_round_hist': {},
            'gdfs_win_index_hist': {},
        })
    summary = {
        key: value for key, value in _LEAPWASTE_AGG_STATE.items()
        if key != 'last_flush'
    }
    if not _GDFS:
        # Keep disabled sweeps byte-identical; the state keys are present so
        # a pid reset always has one complete schema, but empty probe fields
        # must not appear in old aggregate output.
        for key in _GDFS_HIST_KEYS:
            summary.pop(key, None)
    path = os.path.join(_LEAPWASTE_DIR, f'leapagg_{pid}.json')
    temporary = f'{path}.tmp'
    try:
        os.makedirs(_LEAPWASTE_DIR, exist_ok=True)
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, separators=(',', ':'))
        os.replace(temporary, path)
        _LEAPWASTE_AGG_STATE['last_flush'] = now
    except Exception:
        pass


def _leapwaste_aggregate_record(
    frontier_len: int,
    n_successors: int,
    layer: int,
    prefix_formed: bool,
    terminated: bool,
    n_after_win: int | None,
    n_cleared: int | None,
) -> None:
    """Count one iteration without writing a per-iteration record."""
    if not _LEAPWASTE_DIR or not _LEAPWASTE_AGGREGATE:
        return
    _leapwaste_aggregate_flush()
    hist_frontier = _LEAPWASTE_AGG_STATE['frontier_len_hist']
    hist_successors = _LEAPWASTE_AGG_STATE['n_successors_hist']
    hist_layers = _LEAPWASTE_AGG_STATE['layer_hist']
    frontier_key = str(frontier_len)
    successor_key = str(n_successors)
    layer_key = str(layer)
    hist_frontier[frontier_key] = hist_frontier.get(frontier_key, 0) + 1
    hist_successors[successor_key] = hist_successors.get(successor_key, 0) + 1
    hist_layers[layer_key] = hist_layers.get(layer_key, 0) + 1
    _LEAPWASTE_AGG_STATE['active_synth_max_layer'] = max(
        _LEAPWASTE_AGG_STATE['active_synth_max_layer'] or layer,
        layer,
    )
    _LEAPWASTE_AGG_STATE['iterations'] += 1
    if prefix_formed:
        _LEAPWASTE_AGG_STATE['prefix_formed'] += 1
    if terminated:
        _LEAPWASTE_AGG_STATE['terminated'] += 1
    if n_after_win is not None:
        _LEAPWASTE_AGG_STATE['sum_n_after_win'] += n_after_win
        _LEAPWASTE_AGG_STATE['n_after_win_count'] += 1
    if n_cleared is not None:
        _LEAPWASTE_AGG_STATE['sum_n_cleared'] += n_cleared
        _LEAPWASTE_AGG_STATE['n_cleared_count'] += 1
    _leapwaste_aggregate_flush()


def _leapwaste_aggregate_start() -> None:
    """Start one aggregate process state and count its synthesize call."""
    if not _LEAPWASTE_DIR or not _LEAPWASTE_AGGREGATE:
        return
    _leapwaste_aggregate_flush(force=True)
    _LEAPWASTE_AGG_STATE['synth_calls'] += 1
    _LEAPWASTE_AGG_STATE['active_synth_max_layer'] = 0


def _leapwaste_aggregate_finish() -> None:
    """Persist the latest aggregate state at a synthesize boundary."""
    if _LEAPWASTE_AGG_STATE.get('active_synth_max_layer') is not None:
        max_layer = _LEAPWASTE_AGG_STATE['active_synth_max_layer']
        hist = _LEAPWASTE_AGG_STATE['max_layer_per_synth_hist']
        max_layer_key = str(max_layer)
        hist[max_layer_key] = hist.get(max_layer_key, 0) + 1
        _LEAPWASTE_AGG_STATE['active_synth_max_layer'] = None
    _leapwaste_aggregate_flush(force=True)


def _instantiate_single_start(
    circuit: Circuit,
    target: Any,
    seed: int,
    **kwargs: Any,
) -> Circuit:
    """
    Instantiate `circuit` once from one explicitly seeded starting point.

    ``Circuit.instantiate`` with ``multistarts=M`` runs its M starts through
    ``Instantiater.multi_start_instantiate_inplace``, which is a sequential
    list comprehension. Every one of those starts therefore executes inside a
    single worker, so a level-4 compile hides an 8x serial section inside each
    task the search dispatched. This helper is the unit that lets LEAP hand the
    starts to separate workers instead.

    The seed is explicit and per-start rather than inherited from the process.
    ``seed_random_sources`` seeds libc's ``srand``, which Ceres consumes, and
    that state is per-process; running starts on different workers would
    otherwise draw from unrelated streams and make runs irreproducible.
    """
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
        beam_width: int | None = None,
        async_drain: bool = False,
        parallel_multistart: bool = False,
        instantiate_options: dict[str, Any] = {},
        num_prefixes: int = 1,
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

        if beam_width is not None and not is_integer(beam_width):
            raise TypeError(
                'Expected beam_width to be an integer, got %s'
                % type(beam_width),
            )

        if beam_width is not None and beam_width <= 0:
            raise ValueError(
                'Expected beam_width to be positive, got %d.'
                % int(beam_width),
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

        if not is_integer(num_prefixes):
            raise TypeError(
                'Expected num_prefixes to be an integer, got %s'
                % type(num_prefixes),
            )

        if num_prefixes < 1:
            raise ValueError(
                'Expected num_prefixes to be at least 1, got %d.'
                % int(num_prefixes),
            )

        self.num_prefixes = num_prefixes
        self.beam_width = beam_width
        self.async_drain = async_drain
        self.parallel_multistart = parallel_multistart
        self.heuristic_function = heuristic_function
        self.layer_gen = layer_generator
        self.success_threshold = success_threshold
        self.cost = cost
        self.max_layer = max_layer
        self.no_progress_layers_allowed = no_progress_layers_allowed
        self.min_prefix_size = min_prefix_size
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
        leapwaste_enabled = bool(_LEAPWASTE_DIR)
        aggregate_enabled = leapwaste_enabled and _LEAPWASTE_AGGREGATE
        synth_id = _leapwaste_new_synth_id() if leapwaste_enabled else None
        iteration = 0
        if aggregate_enabled:
            _leapwaste_aggregate_start()

        # Initialize run-dependent options
        instantiate_options = self.instantiate_options.copy()

        # Seed the PRNG
        if 'seed' not in instantiate_options:
            instantiate_options['seed'] = data.seed

        # Get layer generator for search
        layer_gen = self._get_layer_gen(data)

        # Begin the search with an initial layer
        frontier = Frontier(utry, self.heuristic_function)
        initial_layer = layer_gen.gen_initial_layer(utry, data)
        initial_layer.instantiate(utry, **instantiate_options)
        frontier.add(initial_layer, 0)

        # Track best circuit, initially the initial layer
        best_dist = self.cost.calc_cost(initial_layer, utry)
        best_circ = initial_layer
        best_layer = 0
        best_dists = [best_dist]

        # Counted for the aggregate probe only. Local, never on self: the
        # pass object is shared across every block ForEachBlockPass
        # dispatches, so per-block state on self would leak one block's
        # trajectory into the next one's.
        tasks_dispatched = 0
        best_layers = [0]
        last_prefix_layer = 0
        previous_popped_id: int | None = None
        previous_popped_layer: int | None = None

        # Track partial solutions
        psols: dict[int, list[tuple[Circuit, float]]] = {}

        _logger.debug(f'Search started, initial layer has cost: {best_dist}.')

        # Evalute initial layer
        if best_dist < self.success_threshold:
            _logger.debug('Successful synthesis with 0 layers.')
            if aggregate_enabled:
                _leapwaste_aggregate_finish()
            return initial_layer

        # Record layers that have been warned about
        # to avoid duplicate warnings
        warned_layers: list[int] = []

        # Main loop
        while not frontier.empty():
            if leapwaste_enabled:
                frontier_len_before_pop = len(frontier)
                n_added_this_iter = 0
                prefix_formed = False
                n_cleared = None
            # Expand enough nodes to fill the task budget, not a fixed count.
            #
            # LEAP pops one node and dispatches its successors, so tasks in
            # flight equal the block's coupling-graph edge count -- measured
            # at 2.07 on sparse devices. That caps how much of a machine one
            # block's search can use, independently of how many workers
            # exist.
            #
            # Two separate budgets control this:
            #
            #   beam_width (B)     how many nodes the frontier may keep.
            #                      A search-quality and memory bound.
            #
            # A resource-driven expansion budget (T) was tried here and
            # removed: E2's bounded_32_96 arm timed out and dropped
            # popped/round from 20.12 to 13.9, so the budget form of T binds
            # earlier than B and costs throughput. T returns in the
            # order-preserving speculation form instead (docs/04 6.1.1b).
            #
            # With beam_width unset this pops exactly one node, as before.
            beam = self.beam_width


            popped = []
            successors = []
            successor_layers = []
            while not frontier.empty():
                top_circuit, top_layer = frontier.pop()

                if _GDFS:
                    popped_id = frontier._last_popped_id
                    descended = (
                        previous_popped_id is not None
                        and frontier._last_popped_parent_id
                        == previous_popped_id
                    )
                    # The first pop of a synthesis has no predecessor, so it is
                    # neither a descent nor a jump. Counting it as a non-descent
                    # would put one spurious zero into every one of the ~29k
                    # synthesis calls -- a systematic bias against the very
                    # quantity this probe exists to measure.
                    if aggregate_enabled and previous_popped_id is not None:
                        descent_hist = _LEAPWASTE_AGG_STATE[
                            'gdfs_descent_hist'
                        ]
                        descent_key = str(int(descended))
                        descent_hist[descent_key] = (
                            descent_hist.get(descent_key, 0) + 1
                        )
                        if (
                            not descended
                            and previous_popped_layer is not None
                        ):
                            jump = previous_popped_layer - top_layer
                            jump_hist = _LEAPWASTE_AGG_STATE[
                                'gdfs_backtrack_jump_hist'
                            ]
                            jump_key = str(jump)
                            jump_hist[jump_key] = (
                                jump_hist.get(jump_key, 0) + 1
                            )
                        post_commit_depth = top_layer - last_prefix_layer
                        post_commit_hist = _LEAPWASTE_AGG_STATE[
                            'gdfs_post_commit_depth_hist'
                        ]
                        post_commit_key = str(post_commit_depth)
                        post_commit_hist[post_commit_key] = (
                            post_commit_hist.get(post_commit_key, 0) + 1
                        )
                    previous_popped_id = popped_id
                    previous_popped_layer = top_layer

                # P0-c. Score this pop against the snapshot taken at the
                # previous pop, then re-snapshot. The snapshot is deliberately
                # taken AFTER popping and BEFORE the children are added: that
                # is exactly the set an ordered speculator would have
                # dispatched alongside this node.
                if _SPEC_PROBE_DIR:
                    # Hold the snapshot for a WINDOW of pops rather than
                    # refreshing every pop. Refreshing made rank 0 almost
                    # tautological -- it only asked "was the frontier minimum
                    # popped", which is true unless a fresh child displaced
                    # it, and produced a curve flat at 87.2% from width 1 to
                    # 128. That measures depth-1 prefetch (in-flight 1 -> 2),
                    # not width-K speculation.
                    #
                    # Width K pays only if several CONSECUTIVE commits come
                    # from ONE dispatch, so the snapshot has to survive them.
                    prev = _SPEC_STATE.get('topk')
                    popped_id = frontier._last_popped_id
                    if prev is not None:
                        age = _SPEC_STATE.get('age', 0)
                        _spec_probe_record(age, prev.get(popped_id))
                        _SPEC_STATE['age'] = age + 1
                    if prev is None or _SPEC_STATE.get('age', 0) >= _SPEC_WINDOW:
                        # id -> rank, so scoring a pop is a dict hit rather
                        # than a scan of a 128-long list once per pop.
                        _SPEC_STATE['topk'] = {
                            eid: rank for rank, eid
                            in enumerate(frontier.topk_ids(_SPEC_PROBE_K))
                        }
                        _SPEC_STATE['age'] = 0

                popped.append((top_circuit, top_layer))
                # The parent's layer must travel with its successors: a
                # shared loop variable would silently mis-record depth for
                # every node after the first, and the search would still
                # look healthy.
                for successor in layer_gen.gen_successors(top_circuit, data):
                    successors.append(successor)
                    successor_layers.append(top_layer)

                if beam is None:
                    # Original behaviour: one node per round.
                    break
                if len(popped) >= beam:
                    # Never speculate on more nodes than the frontier is
                    # allowed to keep. This is the fixed-K expansion of KBFS
                    # (Felner, Kraus & Korf 2003), kept as the ablation
                    # baseline for the order-preserving form.
                    break

            tasks_dispatched += len(successors)
            current_iteration = iteration
            iteration += 1

            # Layer of the first popped node, for logging and probe records
            # that assume a single value per round.
            round_layer = popped[0][1]

            if len(successors) == 0:
                if leapwaste_enabled:
                    _leapwaste_emit({
                        'synth_id': synth_id,
                        'iteration': current_iteration,
                        'layer': round_layer,
                        'n_successors': 0,
                        'map_id': None,
                        't_map_start': None,
                        't_map_end': None,
                        'terminated': False,
                        'win_index': None,
                        'n_after_win': None,
                        'frontier_len_before_pop': frontier_len_before_pop,
                        'frontier_len_after_adds': len(frontier),
                        'n_added_this_iter': n_added_this_iter,
                        'prefix_formed': prefix_formed,
                        'n_cleared': n_cleared,
                    })
                if aggregate_enabled:
                    _leapwaste_aggregate_record(
                        frontier_len_before_pop,
                        0,
                        round_layer,
                        prefix_formed,
                        False,
                        None,
                        n_cleared,
                    )
                continue

            # Instantiate successors
            if leapwaste_enabled:
                t_map_start = time.time()
                map_future = get_runtime().map(
                    Circuit.instantiate,
                    successors,
                    target=utry,
                    **instantiate_options,
                )
                map_id = getattr(map_future, '_bqprof_leapwaste_map_id', None)
                circuits = await map_future
                t_map_end = time.time()
            elif self.async_drain:
                # Consume results as they arrive instead of at a barrier.
                #
                # A barrier makes every round cost the slowest of its tasks.
                # Measured within-batch max/median instantiate duration is
                # 1.31 at the median and 2.94 at p90, so that tail is real,
                # and LEAP pays it once per ~2 tasks. Draining with `next`
                # pays it once per round instead.
                #
                # Results are stored by their index in `successors`, so the
                # evaluation below sees them in the same order a barrier
                # would have produced. Only the waiting changes.
                #
                # Note: the runtime has no wait-any across futures and
                # `RuntimeFuture._done` documents that polling can deadlock,
                # so overlapping the next round's dispatch with this round's
                # tail is not expressible here. Rounds are sized by
                # `beam_width` instead, which amortises the tail over B
                # parents rather than over the block's degree.
                drain_future = get_runtime().map(
                    Circuit.instantiate,
                    successors,
                    target=utry,
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
                # Dispatch every (successor, starting point) pair separately.
                #
                # Otherwise each successor is one task that runs its M starts
                # sequentially inside a single worker. With M=8 at level 4
                # that is an 8x serial section hidden inside every task the
                # search dispatched, and it sits on the cost centre:
                # instantiation is 93.7% of measured instantiate CPU.
                #
                # Starts are independent, so this changes no search semantics
                # and still selects by the same cost function. It does change
                # results: sequentially the starts share one process's random
                # stream, and here each is seeded explicitly instead. Both are
                # deterministic, but they are not the same stream, so this is
                # not the byte-identical change that pipelining assembly was.
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
                    **instantiate_options,
                )

            # Evaluate successors
            for win_index, circuit in enumerate(circuits):
                # Depth of *this* successor's parent, not the round's first.
                layer = successor_layers[win_index]
                dist = self.cost.calc_cost(circuit, utry)

                if dist < self.success_threshold:
                    if _GDFS:
                        solutions_in_round = 1
                        for remaining in circuits[win_index + 1:]:
                            if (
                                self.cost.calc_cost(remaining, utry)
                                < self.success_threshold
                            ):
                                solutions_in_round += 1
                        if aggregate_enabled:
                            solutions_hist = _LEAPWASTE_AGG_STATE[
                                'gdfs_solutions_in_round_hist'
                            ]
                            solutions_key = str(solutions_in_round)
                            solutions_hist[solutions_key] = (
                                solutions_hist.get(solutions_key, 0) + 1
                            )
                            win_hist = _LEAPWASTE_AGG_STATE[
                                'gdfs_win_index_hist'
                            ]
                            win_key = str(win_index)
                            win_hist[win_key] = (
                                win_hist.get(win_key, 0) + 1
                            )
                    _logger.debug(
                        f'Successful synthesis with {layer + 1} layers.',
                    )
                    if self.store_partial_solutions:
                        data['psols'] = psols
                    if leapwaste_enabled:
                        _leapwaste_emit({
                            'synth_id': synth_id,
                            'iteration': current_iteration,
                            'layer': layer,
                            'n_successors': len(successors),
                            'map_id': map_id,
                            't_map_start': t_map_start,
                            't_map_end': t_map_end,
                            'terminated': True,
                            'win_index': win_index,
                            'n_after_win': len(circuits) - win_index - 1,
                            'frontier_len_before_pop': frontier_len_before_pop,
                            'frontier_len_after_adds': len(frontier),
                            'n_added_this_iter': n_added_this_iter,
                            'prefix_formed': prefix_formed,
                            'n_cleared': n_cleared,
                        })
                    if aggregate_enabled:
                        _leapwaste_aggregate_record(
                            frontier_len_before_pop,
                            len(successors),
                            layer,
                            prefix_formed,
                            True,
                            len(circuits) - win_index - 1,
                            n_cleared,
                        )
                        _leapwaste_aggregate_finish()
                    return circuit

                if self.check_new_best(layer + 1, dist, best_layer, best_dist):
                    plural = '' if layer == 0 else 's'
                    _logger.debug(
                        f'New best circuit found with {layer + 1} layer{plural}'
                        f' and cost: {dist:.12e}.',
                    )
                    best_dist = dist
                    best_circ = circuit
                    best_layer = layer + 1

                    if self.check_leap_condition(
                        layer + 1,
                        best_dist,
                        best_layers,
                        best_dists,
                        last_prefix_layer,
                    ):
                        _logger.debug(f'Prefix formed at {layer + 1} layers.')
                        last_prefix_layer = layer + 1
                        if leapwaste_enabled:
                            prefix_formed = True
                            n_cleared = len(frontier)
                        # P0-d: what is being thrown away, and was it worth
                        # keeping? This is the gate on multi-prefix LEAP.
                        # frontier.clear() is a destructive commit -- one
                        # greedy, unbacktrackable choice of continuation --
                        # and lowering min_prefix_size makes that choice more
                        # often, which is the likely reason docs/19's
                        # intervention stalled at 3.5x. Recorded HERE, before
                        # the clear, because afterwards the alternatives are
                        # gone and no later analysis can reconstruct them.
                        _prefix_probe_record(
                            layer + 1, frontier, circuit, len(frontier),
                        )

                        # Multi-prefix: keep the best few continuations
                        # rather than only the one LEAP picked.
                        #
                        # P0-d measured that 85-93% of prefix formations
                        # discard a candidate the frontier itself ranks
                        # CHEAPER than the one kept. That is possible
                        # because check_new_best keeps by depth progress
                        # while the frontier orders by AStarHeuristic, so
                        # the two criteria disagree -- and racing both is
                        # how you avoid having to know which is right.
                        #
                        # Popped destructively, and BEFORE the clear:
                        # Frontier exposes no way to read the circuits
                        # behind topk_ids, so this is the only way to retain
                        # them. At num_prefixes == 1 the loop body never
                        # runs and what follows is exactly the original
                        # clear-then-add.
                        alternates = []
                        for _ in range(self.num_prefixes - 1):
                            if frontier.empty():
                                break
                            alternates.append(frontier.pop())

                        frontier.clear()
                        if self.max_layer is None or layer + 1 < self.max_layer:
                            frontier.add(circuit, layer + 1)
                            if leapwaste_enabled:
                                n_added_this_iter += 1
                            # Re-seeded at the LAYER THEY HELD, not at
                            # layer + 1: an alternate is a sibling of the
                            # kept node, not a child of it, and promoting it
                            # would corrupt every depth statistic downstream.
                            for alt_circuit, alt_layer in alternates:
                                frontier.add(alt_circuit, alt_layer)
                                if leapwaste_enabled:
                                    n_added_this_iter += 1

                if self.store_partial_solutions:
                    if layer not in psols:
                        psols[layer] = []

                    psols[layer].append((circuit.copy(), dist))

                    if len(psols[layer]) > self.partials_per_depth:
                        psols[layer].sort(key=lambda x: x[1])
                        del psols[layer][-1]

                if self.max_layer is None or layer + 1 < self.max_layer:
                    frontier.add(circuit, layer + 1)
                    if leapwaste_enabled:
                        n_added_this_iter += 1

            if leapwaste_enabled:
                _leapwaste_emit({
                    'synth_id': synth_id,
                    'iteration': current_iteration,
                    'layer': round_layer,
                    'n_successors': len(successors),
                    'map_id': map_id,
                    't_map_start': t_map_start,
                    't_map_end': t_map_end,
                    'terminated': False,
                    'win_index': None,
                    'n_after_win': None,
                    'frontier_len_before_pop': frontier_len_before_pop,
                    'frontier_len_after_adds': len(frontier),
                    'n_added_this_iter': n_added_this_iter,
                    'prefix_formed': prefix_formed,
                    'n_cleared': n_cleared,
                    # Quality trajectory. Without this the probe records what
                    # each round COST but never what it BOUGHT, so a per-block
                    # marginal return -- improvement in best cost per unit of
                    # compute -- cannot be reconstructed offline at any price.
                    # That is the quantity a budget-allocation policy would
                    # have to rank blocks by, and the reason such a policy
                    # cannot be evaluated against the data already on disk.
                    'best_dist': best_dist,
                    'best_layer': best_layer,
                    'tasks_dispatched': tasks_dispatched,
                })
            if aggregate_enabled:
                _leapwaste_aggregate_record(
                    frontier_len_before_pop,
                    len(successors),
                    round_layer,
                    prefix_formed,
                    False,
                    None,
                    n_cleared,
                )

            # Bound the frontier by width.
            #
            # LEAP's own bound is `frontier.clear()` on a formed prefix, which
            # fires on an absolute layer threshold (min_prefix_size). Measured
            # at max_synthesis_size 4 that condition fires in 0.04-0.16% of
            # rounds while the frontier reaches a p90 of 3,676, because the
            # depth a generic w-qubit unitary needs grows with 4^w while the
            # threshold does not. A width bound does not depend on depth.
            # No-op when `beam_width` is unset.
            n_pruned_this_round = frontier.prune(beam)
            if aggregate_enabled and _LEAPWASTE_AGG_STATE:
                _LEAPWASTE_AGG_STATE['sum_n_pruned'] += n_pruned_this_round
                if n_pruned_this_round > 0:
                    _LEAPWASTE_AGG_STATE['rounds_where_prune_fired'] += 1
                _LEAPWASTE_AGG_STATE['sum_popped_per_round'] += len(popped)
                _LEAPWASTE_AGG_STATE['n_rounds'] += 1

            layer_diff = abs(best_layer - round_layer)
            if (
                layer_diff % self.no_progress_layers_allowed == 0
                and layer_diff > 0
                and round_layer not in warned_layers
            ):
                _logger.warning(
                    'No improvement after '
                    f'{self.no_progress_layers_allowed} layers.',
                )
                warned_layers.append(round_layer)

        _logger.warning('Frontier emptied.')
        _logger.warning(
            'Returning best known circuit with %d layer%s and cost: %e.'
            % (best_layer, '' if best_layer == 1 else 's', best_dist),
        )
        if self.store_partial_solutions:
            data['psols'] = psols

        if aggregate_enabled:
            _leapwaste_aggregate_finish()
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
        return delta < 0 and layers_added >= self.min_prefix_size

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
