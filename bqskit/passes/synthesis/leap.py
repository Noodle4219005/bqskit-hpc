"""This module implements the LEAPSynthesisPass."""
from __future__ import annotations

import hashlib
import json
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
from bqskit.runtime.task import PRIORITY_CRITICAL
from bqskit.runtime.task import PRIORITY_SPECULATIVE
from bqskit.runtime.future import RuntimeFuture
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
_LTRACE_DIR = os.environ.get('BQPROF_LTRACE')

# One JSON record per synthesis, written at every exit.
#
# The aggregate probe writes leapagg_<pid>.json, which sums over every block a
# worker touched, so in a whole-circuit run it cannot say WHICH block consumed
# the time. That is the first question to ask about a compute-bound timeout --
# a few pathological blocks and a uniform cost need opposite fixes -- and it
# was unanswerable. This record is per synthesis, so it can.
#
# It also carries the best-distance trajectory, which answers the question that
# decides whether a timing-out block is worth more compute at all: a distance
# still descending when the budget ran out is a budget problem, a distance that
# plateaued is a stuck search, and no amount of parallelism fixes the second.
_BLOCKPROF_DIR = os.environ.get('BQPROF_BLOCKPROF')

# Measure how much of the frontier is the SAME circuit reached by a different
# gate order, before deciding whether merging them is worth building.
#
# LEAP appends one two-qubit block per layer, so a node is a SEQUENCE of gate
# locations. Two sequences that differ only by swapping adjacent gates on
# disjoint qubits describe the same circuit: they commute, so they span exactly
# the same set of reachable unitaries. Exploring both is provably wasted work,
# and unlike a heuristic prune, merging them cannot lose a solution.
#
# The canonical representative is the ASAP schedule -- push every gate as early
# as its qubits allow, then sort within each time step. Two members of a
# commutation class always produce the same ASAP form.
_DUPPROBE_DIR = os.environ.get('BQPROF_DUPPROBE')

# Give critical work the front of the queue by emptying it first.
#
# On by default because the measured alternative is a 12% regression against
# stock on a saturated machine. Set BQSKIT_SPEC_YIELD=0 to reproduce the old
# dispatch-order-only behaviour.
# Cancel every in-flight speculation before dispatching critical work.
#
# This existed because the worker's ready queue was a FIFO: once speculation
# was queued, critical work could only wait behind it, and the only remedy was
# to empty the queue. The comment at the cancellation site still says exactly
# that -- "since the queue cannot be reordered, it is emptied instead".
#
# The queue CAN be reordered now. The QoS service classes put critical work
# ahead of speculation regardless of arrival order, which is the same guarantee
# without destroying the work: measured, cancellation threw away 67-81% of all
# speculative tasks, and because it fires every round, speculation never
# survives longer than one round's gap. That is why occupancy could not climb
# even when a single block had the whole machine -- the mechanism meant to
# protect the critical path was also the mechanism capping utilisation.
#
# So it defaults to the inverse of QoS: it is the fallback for a runtime whose
# queue cannot be reordered, and nothing more. BQSKIT_SPEC_YIELD still forces
# it either way for A/B.
_SPEC_YIELD_ENV = os.environ.get('BQSKIT_SPEC_YIELD')

_TASKLOG_DIR = os.environ.get('BQSKIT_TASKLOG_DIR')
"""Same switch as the worker probe; unset disables memo events."""

# How old an occupancy reading may be before it is treated as unknown. Set to
# several broadcast intervals: one missed broadcast is normal jitter, but a
# reading from seconds ago describes a machine that has since emptied or
# filled, and acting on that is worse than falling back to the estimator.
_OCCUPANCY_STALE_AFTER = 1.0

# Speculation value bound. Below the floor a speculative task is worth less
# than whatever it displaces, so K stops growing there regardless of how many
# cores are free. Warmup exists because a hit rate estimated from three samples
# would swing K wildly early in a synthesis.
_SPEC_VALUE_FLOOR = float(os.environ.get('BQSKIT_SPEC_VALUE_FLOOR', '0.02'))
_SPEC_VALUE_WARMUP = int(os.environ.get('BQSKIT_SPEC_VALUE_WARMUP', '32'))

# Scheduling counters that are not memo events but must still reach the task-log
# event stream. Without this, `stall_throttled_rounds` only ever landed in the
# BQPROF_LEAPWASTE_AGGREGATE aggregate, so a run that did not enable it could not
# tell whether the throttle had fired at all -- which is the state the tail
# investigation of 2026-08-13 found itself in.
_SCHEDULE_METRICS = frozenset({
    'stall_throttled_rounds',
    'stall_yield_suppressed',
    'width_from_measured',
    'width_from_estimator',
    'value_capped',
})

# Service class actually used for speculation. Set BQSKIT_TASK_QOS=0 to submit
# speculation as PRIORITY_CRITICAL, which makes the worker's priority queue
# degenerate to the FIFO it replaced.
#
# This exists so an A/B differs in exactly one variable. Comparing this tree
# against an older snapshot would also change fair-share, the occupancy
# broadcast and every leap.py edit since -- and an A/B with six differences
# cannot attribute anything.
_TASK_QOS = os.environ.get('BQSKIT_TASK_QOS', '1') != '0'
_SPEC_PRIORITY = PRIORITY_SPECULATIVE if _TASK_QOS else PRIORITY_CRITICAL

# Resolved here because it depends on _TASK_QOS: yield is what a FIFO runtime
# has to do, and QoS replaces it.
_SPEC_YIELD = (
    (_SPEC_YIELD_ENV != '0') if _SPEC_YIELD_ENV is not None else not _TASK_QOS
)


def _commutation_canon(circuit: Circuit) -> tuple[Any, ...]:
    """Canonical form of a circuit's two-qudit gate order, modulo commutation."""
    ready: dict[int, int] = {}
    steps: dict[int, list[tuple[int, ...]]] = {}
    for op in circuit:
        if op.num_qudits < 2:
            continue
        loc = tuple(sorted(int(q) for q in op.location))
        step = max((ready.get(q, 0) for q in loc), default=0)
        steps.setdefault(step, []).append(loc)
        for q in loc:
            ready[q] = step + 1
    return tuple(tuple(sorted(steps[s])) for s in sorted(steps))


_CircuitStructureKey = tuple[
    tuple[Gate, CircuitLocation, tuple[float, ...]],
    ...,
]


class _SpeculationFlight(NamedTuple):
    """One dispatched speculation batch that has not been harvested yet.

    Several may be outstanding at once: a batch is refilled as workers free up
    rather than at batch boundaries, so the flights overlap.
    """

    future: Any
    keys: list[Any]
    batches: list[list[Circuit]]
    owners: list[tuple[int, int]] | None
    epoch: int
    bound_generation: int | None
    n_tasks: int


class _SpeculationMemoEntry(NamedTuple):
    """One deferred expansion and the logical context that produced it."""

    epoch: int
    bound_generation: int | None
    successors: list[Circuit]
    results: list[Circuit]
    used: bool


def _logical_trace(record: dict[str, Any]) -> None:
    """Append one logical-search event when the trace probe is enabled."""
    if not _LTRACE_DIR:
        return
    os.makedirs(_LTRACE_DIR, exist_ok=True)
    path = os.path.join(_LTRACE_DIR, f'ltrace_{os.getpid()}.jsonl')
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, separators=(',', ':')) + '\n')


def _prefix_probe_record(
    layer: int,
    frontier: Any,
    kept: Any,
    n_frontier: int,
) -> None:
    """Record the cost of the kept continuation against the discarded ones.

    The multi-prefix mechanism it motivated was measured and removed; see
    results/prefix_diversity/summary_run_1018476.txt for the 91.8% / 95.7%
    figures.
    """
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
            'n_rollbacks': 0,
            'sum_backjump_depth': 0,
            'stack_depth_at_rollback': {},
            'depth_burned_hist': {},
            'n_rollback_rescued': 0,
            'n_exhausted_unverified': 0,
            'deepen_raises': 0,
            'deepen_solved_after_raise': 0,
            'deepen_overflow_max': 0,
            'frontier_len_hist': {},
            'n_successors_hist': {},
            'layer_hist': {},
            'max_layer_per_synth_hist': {},
            'active_synth_max_layer': None,
            'spec_hits': 0,
            'spec_misses': 0,
            'spec_evicted': 0,
            'spec_unused': 0,
            'spec_eventual_hits': 0,
            'spec_timely_hits': 0,
            'spec_late': 0,
            'spec_stale_epoch': 0,
            # --- HPC accounting ---
            # critical_stall_s is the metric speculation actually exists to
            # minimise: seconds the logical search spent blocked on results it
            # cannot proceed without. Wall clock alone cannot separate "the
            # search got faster" from "the search waited less"; this can.
            'critical_stall_s': 0.0,
            # Work split. Dispatched tasks are also the communication volume:
            # every task ships a Circuit to a worker and receives one back, so
            # these counts are message counts, and the op sums are a payload
            # proxy for how much circuit was actually moved.
            'critical_tasks': 0,
            'spec_tasks': 0,
            'critical_payload_ops': 0,
            'spec_payload_ops': 0,
            'dispatch_rounds': 0,
            # Which width source actually decided K, per round. The occupancy
            # broadcast is silent when absent -- a run with no manager, or with
            # a stale reading, falls back to the estimator and looks identical
            # from the outside. Without these two counters an A/B cannot tell
            # "the measured path did not help" from "the measured path never
            # ran", and those have opposite conclusions.
            'width_from_measured': 0,
            'width_from_estimator': 0,
            'idle_seen_sum': 0,
            # Memory high-water, recorded rather than bounded.
            'cache_peak_entries': 0,
            'cache_peak_results': 0,
            # Contention: mean over rounds of (batch wall / fastest batch
            # wall). ~1 means this block had the machine to itself; ~N means it
            # was sharing with about N others.
            'contention_sum': 0.0,
            'contention_samples': 0,
            'stall_throttled_rounds': 0,
            # The other two arms of the same two counters. Both were emitted by
            # `record_spec_metric` while missing from THIS dict, and because the
            # increment below was a bare `+=` on a plain dict that is a
            # KeyError, not a lost sample: job 1024323 (bigm, four arms) ran for
            # 2 h 17 m and every arm died inside `synthesize` on
            # `KeyError: 'value_capped'`. The job exited COMPLETED with no
            # result.json, so it read as "the circuit was too big" rather than
            # as a crash. `stall_yield_suppressed` was the same bug waiting on
            # a run that both enabled the aggregate and took the non-scarce
            # branch. Declaring them is the narrow fix; the guard in
            # `record_spec_metric` is the one that stops it recurring.
            'value_capped': 0,
            'stall_yield_suppressed': 0,
            # Tasks thrown away to clear the queue for critical work, and how
            # many rounds did it. Together they price the guarantee: if
            # cancellation is large relative to spec_tasks, speculation is
            # being issued faster than it can pay off.
            'spec_cancelled': 0,
            'spec_yield_rounds': 0,
            # How many separate speculation batches were dispatched. With the
            # old one-at-a-time rule this equalled the number of rounds that
            # dispatched at all; a higher count means refill is working.
            'spec_flights': 0,
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

        # K is a resource knob, not a search knob: it decides how far ahead
        # speculation runs, never which node is popped. 'auto' sizes it from
        # the worker pool so speculation cannot displace critical work --
        # with s successors per node and R = s reserving room for one critical
        # batch, holding (K-1)*s <= W - R gives
        #
        #     K <= 1 + (W - s)/s
        #
        # so "speculation must never delay mandatory work" is a property of
        # the construction rather than something to hope for. Any fixed
        # integer keeps the previous behaviour and is the ablation baseline.

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

        # W. This pass runs inside a worker and cannot see the Compiler's
        # num_workers, so the width has to be supplied rather than discovered.
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

        # M: the speculative cache budget. Its owner is system memory, not the
        # search -- evicting an entry only throws away computation, the logical
        # node is still in the frontier and can be recomputed, so M cannot
        # change what the search does, only what it costs.
        #
        # The previous default of 4*K was far too small and made eviction the
        # common case rather than the backstop: measured over the qv20/tokyo
        # sweep it discarded MORE than it served (spec_evicted 48-83 against
        # spec_hits 52-82). That is the wrong trade in this domain, where a
        # node costs 491 ms to evaluate and 1,768 bytes to store -- one GB of
        # stored results stands for 77 CPU-hours of numerical optimisation, so
        # reclaiming memory here buys a resource that is not scarce with one
        # that is.
        #
        # The cap exists only to stop a runaway, not to save memory.
        #
        # It was 8, which was chosen when the memo was sized from K and memory
        # was treated as scarce. It is not: a node costs 491 ms to evaluate and
        # 1,768 bytes to store, so a gigabyte of cached results stands for 77
        # CPU-hours of numerical optimisation. Bounding K to protect memory
        # traded the resource that is scarce for the one that is not.
        #
        # 8 was also measured to be BINDING rather than protective: with s = 3
        # successors on a sparse 4-qubit block and W = 96, the resource formula
        # asks for K = 1 + (96-3)/3 = 32, so the cap was discarding three
        # quarters of the parallelism the machine could have absorbed. The
        # default was W itself, which still prevented a useful backlog.
        # The former W default limited speculative depth to no more than the
        # workers assigned to this block: it intentionally avoided backlog.
        # Idle workers create no value, while a backlog is what lets cost
        # ordering help. Even K=32 used only 23.5% of 112 cores during gate
        # removal, so begin at 4W; BQSKIT_EXPAND_K_MAX remains an override.
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

        # Sizing the cache from K was the mistake: it made the cheap resource
        # (memory) rationed by the expensive one (parallelism). The default is
        # now effectively unbounded within a synthesis -- `cache_peak_entries`
        # is recorded instead, so the real high-water mark is a measurement
        # rather than a guess, and `spec_evicted` should stay 0.
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
        # The brake's threshold is an absolute constant -- compile.py:1025
        # passes [3, 4, 7, 9][level - 1] -- and nothing about it scales with
        # how deep the search can actually go. Measured, that makes it
        # unreachable: at msz=3 only 2.02% of synthesis calls ever reach layer
        # 7, so prefix_formed is 0 across 15 stock runs and the brake is dead
        # code. E7 then showed the consequence for anything built on top of
        # it: with max_layer=4 (below the threshold) no prefix can form, so a
        # rollback has nothing to return to and 309 of 325 syntheses ended in
        # the unverified exit; with max_layer=8 (above it) prefixes formed and
        # 2-3 rollbacks removed those exits entirely.
        #
        # So the threshold has to be relative to the depth that is available,
        # not a constant. When a depth bound is in force the bound IS that
        # depth, and the fraction below derives the threshold from it. Unset
        # leaves the absolute constant exactly as it was.
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

        def dispatch_batches(
            batches: list[list[Circuit]],
            task_priority: int = PRIORITY_CRITICAL,
        ) -> tuple[RuntimeFuture, list[tuple[int, int]] | None]:
            """Dispatch several node expansions in one runtime map.

            `task_priority` is the worker service class. Speculation passes
            PRIORITY_SPECULATIVE so it can go as deep as memory allows without
            ever standing in front of the work the answer waits on: the ready
            queue is ordered by (class, arrival), so a critical batch submitted
            after a thousand speculative ones still runs next.

            Without this the two are indistinguishable once queued, and the
            only lever left is how little to speculate -- which is what capped
            occupancy at 59.2% while LEAP was active.
            """
            flat_successors = [
                successor
                for batch in batches
                for successor in batch
            ]

            if (
                not leapwaste_enabled
                and not self.async_drain
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

        def record_spec_metric(metric: str, amount: float = 1) -> None:
            """Record ordered-speculation accounting when aggregation is on.

            When BQSKIT_TASKLOG_DIR is set this also emits a timestamped event,
            so the memo's contents can be replayed rather than summarised. The
            counters say how many hits there were; the event stream says *when*
            each entry was computed, when it was used, and when it went stale --
            which is the only way to see whether a speculative result arrived
            before the critical path needed it.
            """
            if aggregate_enabled:
                # `+=` on a plain dict raises KeyError for a metric that was
                # never declared in the aggregate's key set, and that exception
                # propagates out of `synthesize` and kills the compile. A probe
                # must never be able to do that. Job 1024323 lost four arms and
                # 2 h 17 m of a 448-core allocation to exactly this, and its
                # Slurm state was COMPLETED, so nothing about the failure said
                # "probe".
                #
                # Self-registering instead of raising costs one thing: a typo in
                # a metric name silently becomes a new counter rather than an
                # error. That is why the unknown name is ALSO appended to
                # `undeclared_metrics`, which lands in the aggregate file -- a
                # typo shows up as a named entry there instead of as a crash or
                # as silence. This project has been bitten more often by probes
                # that reported confidently on nothing than by typos.
                if metric in _LEAPWASTE_AGG_STATE:
                    _LEAPWASTE_AGG_STATE[metric] += amount
                else:
                    _LEAPWASTE_AGG_STATE[metric] = amount
                    unknown = _LEAPWASTE_AGG_STATE.setdefault(
                        'undeclared_metrics', [],
                    )
                    if metric not in unknown:
                        unknown.append(metric)
            if _memo_ev is not None and (
                metric.startswith('spec_') or metric in _SCHEDULE_METRICS
            ):
                _memo_ev(metric, len(speculation_memo))

        # Per-block memo event stream. One line per memo transition, with the
        # memo's size at that moment, so a replay can show what the cache held
        # rather than only how many times it was hit.
        _memo_ev = None
        if _TASKLOG_DIR:
            _blk = '%x' % (abs(hash(utry)) & 0xFFFFFF)
            _mf = open(
                os.path.join(_TASKLOG_DIR, f'memo_{os.getpid()}.jsonl'),
                'a', buffering=1,   # line-buffered: see below
            )

            def _memo_ev(ev: str, size: int) -> None:
                _mf.write(
                    '{"t":%.6f,"blk":"%s","ev":"%s","n":%d}\n'
                    % (time.time(), _blk, ev, size)
                )

        speculation_memo: dict[
            _CircuitStructureKey,
            _SpeculationMemoEntry,
        ] = {}
        speculation_flights: list[_SpeculationFlight] = []
        epoch = 0
        # s: successors per node, measured rather than assumed. None until the
        # first expansion has been seen, which is why round one runs serially.
        succ_ema: float | None = None
        _bp_start = time.perf_counter()
        _bp_traj: list[tuple[float, int, float]] = []
        # Stall detector. A block that has stopped improving is still paying
        # for speculation, and on a shared pool that speculation is taken from
        # blocks that ARE still improving. Measured on a 20-qubit msz=4
        # compile: the six 4-qubit blocks consumed 90% of the time, and one of
        # them spent 334 of its 496 seconds after its last improvement.
        #
        # The gap between improvements is not constant, so a fixed round count
        # cannot tell "searching" from "stuck". Comparing the current gap to
        # this block's OWN typical gap can, and needs no global state: each
        # block throttles itself, and the pool redistributes automatically
        # because a block that dispatches less leaves room in the queue.
        _last_improve_round = 0
        _gap_ema: float | None = None
        # Running speculation value, used to cap K by worth rather than by
        # capacity. See the value bound where effective_k is computed.
        _spec_issued = 0
        _spec_hits = 0
        _dup_seen: set[Any] = set()
        # The set that matters for savings: canonical forms already SENT TO
        # instantiate. add_child sees nodes after the expensive part is
        # already paid, so its duplicate rate is an upper bound on waste, not
        # a saving. `_commutation_canon` reads only gate locations, never
        # parameters, so it can be evaluated on a bare successor template --
        # before instantiate, where skipping actually costs nothing.
        _dup_pre_seen: set[Any] = set()

        def note_pre_dup(successors: list[Circuit]) -> list[Circuit]:
            """Count commutation duplicates for instrumentation."""
            if _DUPPROBE_DIR is None:
                return successors
            for _c in successors:
                _dup_state['pre_generated'] += 1
                _k = _commutation_canon(_c)
                # Global: measurement only, never a control decision.
                if _k in _dup_pre_seen:
                    _dup_state['pre_dup'] += 1
                else:
                    _dup_pre_seen.add(_k)
            return successors
        _dup_state: dict[str, Any] = {
            'added': 0, 'dup': 0, 'by_layer': {}, 'dup_by_layer': {},
            'pre_generated': 0, 'pre_dup': 0,
        }

        def emit_blockprof(status: str, layer: Any, dist: Any) -> None:
            """Write this synthesis's cost and best-distance trajectory."""
            if not _BLOCKPROF_DIR:
                return
            try:
                os.makedirs(_BLOCKPROF_DIR, exist_ok=True)
                digest = hashlib.blake2b(
                    np.ascontiguousarray(
                        np.round(np.asarray(utry), 12),
                    ).tobytes(), digest_size=8,
                ).hexdigest()
                path = os.path.join(
                    _BLOCKPROF_DIR, f'blockprof_{os.getpid()}.jsonl',
                )
                with open(path, 'a', encoding='utf-8') as fh:
                    fh.write(json.dumps({
                        'target': digest,
                        'num_qudits': int(utry.num_qudits),
                        'status': status,
                        'wall_s': round(time.perf_counter() - _bp_start, 4),
                        'final_layer': layer,
                        'final_dist': None if dist is None else float(dist),
                        'trajectory': [
                            [round(t, 4), int(l), float(d)] for t, l, d in _bp_traj
                        ],
                        'pre_generated': _dup_state['pre_generated'],
                        'pre_dup': _dup_state['pre_dup'],
                        'dup_added': _dup_state['added'],
                        'dup_hits': _dup_state['dup'],
                        'dup_by_layer': _dup_state['dup_by_layer'],
                        'added_by_layer': _dup_state['by_layer'],
                    }, separators=(',', ':')) + '\n')
            except Exception:
                pass

        # Contention estimator. `fastest_batch_s` is the uncontended reference:
        # the one round that got the machine to itself. Everything slower than
        # it is other blocks competing for the same workers.
        fastest_batch_s: float | None = None
        contention_ema: float = 1.0

        def circuit_structure_key(circuit: Circuit) -> _CircuitStructureKey:
            """Return an exact identity for one instantiation input.

            The parameters are part of the key, not decoration. Successors are
            built as `circuit.copy()` plus one appended layer, so a node
            carries its parent's optimised parameters. Two nodes reached by
            different paths can share a gate/location sequence while holding
            different parameters -- and they then settle into DIFFERENT local
            minima.

            Measured on a structure that cannot represent its target (one CNOT
            against a random SU(4), which needs three), four starting points
            gave four distinct costs:

                5.3206551509e-02  5.3206544435e-02
                5.3206507129e-02  5.3206521624e-02

            A structure-only key therefore lets a cache hit return a node with
            a different cost, which reorders the frontier. That is speculation
            changing commitment -- exactly what the design forbids. Note that
            bit-identity checks pass with the unsound key whenever the run
            happens not to collide, so they cannot be the guard here.

            CORRECTED 2026-08-14. This docstring used to continue "and
            instantiate starts from them", and that is FALSE, unconditionally.
            `Circuit.instantiate` has one
            exit, `Instantiater.multi_start_instantiate_inplace`, and all four
            of its implementations hardcoded `RandomStartGenerator()` -- so the
            carried parameters were discarded at every multistarts value,
            including 1. Probe: a circuit whose parameters WERE the exact
            solution still got a start 4.7668 rad away and converged elsewhere.
            The four costs quoted above are evidence OF random starts, not of
            inherited ones; a true measurement was attached to a false
            mechanism, which reads more convincingly than a guess would have.

            The conclusion survives the correction, so the key is unchanged:
            including parameters is now permanently over-strict while starts
            are random (fewer hits, never a wrong hit), kept only because
            removing them is a separate unverified change.

            Keying on (gates, locations, parameters) loses nothing the cache
            was for: a node before and after a commit is the SAME circuit with
            the SAME parameters, so it still hits across commits and
            rollbacks. What it stops sharing is two genuinely different
            computations that merely look alike.
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
            """Insert one expansion, evicting the oldest entry if over budget.

            Eviction is a memory backstop, never a search decision: the
            evicted entry's logical node is untouched and will simply be
            recomputed if it is ever popped. `spec_evicted` should be 0 in a
            healthy run -- a nonzero count means M is binding and is costing
            recomputation.
            """
            while (
                structure_key not in speculation_memo
                and len(speculation_memo) >= self.speculate_memo
            ):
                oldest_key = next(iter(speculation_memo))
                speculation_memo.pop(oldest_key)
                record_spec_metric('spec_evicted')
            if aggregate_enabled:
                _LEAPWASTE_AGG_STATE['cache_peak_entries'] = max(
                    _LEAPWASTE_AGG_STATE['cache_peak_entries'],
                    len(speculation_memo) + 1,
                )
                _LEAPWASTE_AGG_STATE['cache_peak_results'] = max(
                    _LEAPWASTE_AGG_STATE['cache_peak_results'],
                    sum(len(e.results) for e in speculation_memo.values()),
                )
            if _memo_ev is not None:
                _memo_ev('spec_insert', len(speculation_memo) + 1)
            speculation_memo[structure_key] = _SpeculationMemoEntry(
                dispatch_epoch,
                bound_generation,
                *expansion,
                False,
            )

        def finish_speculation() -> None:
            """Account for and stop speculative work at synthesis exit."""
            nonlocal speculation_flights
            record_spec_metric(
                'spec_unused',
                sum(not entry.used for entry in speculation_memo.values())
                + sum(len(f.keys) for f in speculation_flights),
            )
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
        deepen_raised = False

        def add_child(circuit: Circuit, layer: int) -> bool:
            """Add a child to the frontier or retain it past the bound."""
            if _DUPPROBE_DIR is not None:
                _dup_state['added'] += 1
                _key = _commutation_canon(circuit)
                if _key in _dup_seen:
                    _dup_state['dup'] += 1
                    _dup_state['dup_by_layer'][layer] = (
                        _dup_state['dup_by_layer'].get(layer, 0) + 1
                    )
                else:
                    _dup_seen.add(_key)
                _dup_state['by_layer'][layer] = (
                    _dup_state['by_layer'].get(layer, 0) + 1
                )
            if current_max_layer is None or layer < current_max_layer:
                frontier.add(circuit, layer)
                return True
            if overflow is not None:
                overflow.append((circuit, layer))
                if aggregate_enabled:
                    _LEAPWASTE_AGG_STATE['deepen_overflow_max'] = max(
                        _LEAPWASTE_AGG_STATE['deepen_overflow_max'],
                        len(overflow),
                    )
            return False

        # Begin the search with an initial layer
        frontier = Frontier(utry, self.heuristic_function)
        initial_layer = layer_gen.gen_initial_layer(utry, data)
        prefetched_initial_successors: list[Circuit] | None = None
        prefetched_initial_future: RuntimeFuture | None = None
        prefetched_initial_owners: list[tuple[int, int]] | None = None
        if self.parallel_multistart and int(
            instantiate_options.get('multistarts', 1),
        ) > 1:
            # The initial layer is one circuit per block, so its only
            # parallelism is its M independent starts. At msz=6 on
            # square_heisenberg_N16 there are six blocks: this map is 6 x M
            # tasks (24 at M=4). More importantly, the first expansion is
            # certain because the initial layer is the frontier's only node.
            # Its 6 x C(6, 2) x M = 360 tasks at M=4 fill the first trough
            # against 96 workers, where measured msz=6 occupancy was median
            # 9.0 with p90 86.7: bursty capacity that this work can use without
            # speculation or a T_K = T_1 argument.
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
            prefetched_initial_successors = note_pre_dup(list(
                layer_gen.gen_successors(initial_layer, data),
            ))
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
        n_rollbacks = 0
        previous_popped_id: int | None = None
        previous_popped_layer: int | None = None

        # Track partial solutions
        psols: dict[int, list[tuple[Circuit, float]]] = {}

        _logger.debug(f'Search started, initial layer has cost: {best_dist}.')

        # Evalute initial layer
        if best_dist < self.success_threshold:
            _logger.debug('Successful synthesis with 0 layers.')
            _logical_trace({
                'event': 'success', 'layer': 0, 'distance': float(best_dist),
            })
            if aggregate_enabled:
                _leapwaste_aggregate_finish()
            emit_blockprof('success_layer0', 0, float(best_dist))
            abandon_prefetched_initial()
            return initial_layer

        # Record layers that have been warned about
        # to avoid duplicate warnings
        warned_layers: list[int] = []

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
                    if aggregate_enabled:
                        # Is there a choice to make at all? 621 rollbacks
                        # produced a total backjump depth of 23, i.e. 96%
                        # chronological -- and that is consistent with two
                        # very different worlds: regret is a poor criterion,
                        # or the stack holds one entry and argmax has nothing
                        # to pick between. Commits and rollbacks run nearly
                        # 1:1 (348 against 336 at max_layer=4), which points
                        # at the second. Measure it rather than argue it.
                        _hist = _LEAPWASTE_AGG_STATE['stack_depth_at_rollback']
                        _key = str(len(_states))
                        _hist[_key] = _hist.get(_key, 0) + 1
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
                    _logical_trace({
                        'event': 'rollback',
                        'index': _target,
                        'depth': frontier.committed_depth(),
                    })
                    if aggregate_enabled:
                        # What the depth-burned criterion WOULD have seen, so
                        # the two can be compared without switching yet.
                        _tl = restored_state.get('layer') if isinstance(
                            restored_state, dict,
                        ) else None
                        if _tl is not None and previous_popped_layer is not None:
                            _burn = max(0, previous_popped_layer - _tl)
                            _bh = _LEAPWASTE_AGG_STATE['depth_burned_hist']
                            _bk = str(_burn)
                            _bh[_bk] = _bh.get(_bk, 0) + 1
                        _LEAPWASTE_AGG_STATE['sum_backjump_depth'] += (
                            len(_states) - 1 - _target
                        )
                    if aggregate_enabled:
                        _LEAPWASTE_AGG_STATE['n_rollbacks'] += 1
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
                    _logical_trace({
                        'event': 'deepen',
                        'old_bound': old_max_layer,
                        'new_bound': current_max_layer,
                    })
                    for circuit, layer in overflow:
                        frontier.add(circuit, layer)
                    overflow.clear()
                    deepen_raised = True
                    if aggregate_enabled:
                        _LEAPWASTE_AGG_STATE['deepen_raises'] += 1
                    continue
                break

            if leapwaste_enabled:
                frontier_len_before_pop = len(frontier)
                n_added_this_iter = 0
                prefix_formed = False
                n_cleared = None
            # Read once per round, used twice: to size K below, and to decide
            # whether the stall throttle has anyone to yield to. None means the
            # occupancy broadcast is missing or stale.
            measured_idle: int | None = None

            # K is recomputed every round from the measured successor width.
            # It only decides how many frontier nodes are speculated on; the
            # pop below is untouched by it, so any K schedule -- including one
            # that changes every round -- leaves the logical trace alone. That
            # is what makes a resource-driven K legitimate rather than a
            # second search parameter.
            if self.expand_k_auto:
                if succ_ema is None:
                    # Nothing measured yet. Run serially rather than guess a
                    # width and over-commit the pool on the first round.
                    effective_k = 1
                else:
                    s = max(1, round(succ_ema))
                    # W is not the pool, it is this block's SHARE of the pool.
                    # ForEachBlockPass dispatches every block at once, so a
                    # width taken from the whole machine is claimed N times
                    # over and speculation starts displacing other blocks'
                    # critical work -- the very thing the resource rule exists
                    # to prevent.
                    #
                    # Prefer the measured idle count over any estimate. The
                    # contention estimator below divides by "this batch versus
                    # the fastest batch this block has seen", and that is blind
                    # in the one case it has to get right: a block that has
                    # been contended since its first round has a contended
                    # fastest, so the ratio sits at 1 and it never throttles.
                    # An absolute reading of how many workers are free right
                    # now cannot make that mistake.
                    #
                    # This also makes the two mechanisms compose. Reuse is
                    # always on because it pays in memory, and memory is
                    # absurdly cheap here -- 1 GB of stored nodes is worth 77
                    # CPU-hours of recomputation. Speculation then tops up
                    # whatever idle capacity reuse could not fill, and the
                    # formula degenerates on its own: saturated means idle ~ 0
                    # means k = 1 means no speculation. No threshold, no
                    # policy switch, no hysteresis to tune.
                    measured_idle = self._measured_idle_workers()
                    if measured_idle is not None:
                        share = max(float(s), float(measured_idle))
                        record_spec_metric('width_from_measured')
                        record_spec_metric('idle_seen_sum', measured_idle)
                    else:
                        share = max(
                            float(s),
                            self.worker_width / max(1.0, contention_ema),
                        )
                        record_spec_metric('width_from_estimator')

                    # Cap K by what speculation is WORTH, not only by what the
                    # machine can hold.
                    #
                    # Capacity alone is an unbounded rule: the frontier always
                    # offers another node, so a worker that runs out of
                    # critical work will always find speculation to do, will
                    # never report itself idle, and the machine will read 100%
                    # busy while measured hit rate is 6.3% -- 94% of that
                    # occupancy is waste. Occupancy becomes a vanity metric the
                    # moment the work filling it is unbounded.
                    #
                    # Value decays geometrically in depth while cost grows
                    # linearly: reaching depth d needs the search to follow the
                    # predicted path d times, so the payoff is ~p^d against a
                    # cost of ~d*s*M. An optimum therefore exists and it is
                    # small, which means filling every core with speculation is
                    # provably past it.
                    #
                    # p is estimated from this block's own timely hits. Timely,
                    # not eventual: only a hit that arrives before the critical
                    # path needs it has shortened anything.
                    if _spec_issued >= _SPEC_VALUE_WARMUP:
                        p = _spec_hits / _spec_issued
                        if p <= 0.0:
                            value_k = 2.0
                        elif p >= 1.0:
                            # Every speculative task has hit, so there is
                            # nothing to discount and the cap must not bind.
                            # log(1.0) is 0, which made this a
                            # ZeroDivisionError -- and only on arms with
                            # speculation enabled, so the control arm passed
                            # and the crash surfaced as the runtime's generic
                            # 'Server connection unexpectedly closed'.
                            value_k = float(self.expand_k_max)
                        else:
                            # Largest d with p^d above the floor; the floor is
                            # the point below which a speculative task is worth
                            # less than the critical task it displaces.
                            value_k = 1.0 + math.log(
                                _SPEC_VALUE_FLOOR,
                            ) / math.log(p)
                        share = min(share, max(float(s), value_k * s))
                        record_spec_metric('value_capped')
                    effective_k = max(
                        1,
                        min(
                            self.expand_k_max,
                            1 + int((share - s) // s),
                        ),
                    )
            else:
                effective_k = self.expand_k

            # Give the pool back when this block stops making progress.
            #
            # Speculation is a bet that the search will keep moving along the
            # frontier. Once a block has gone several of its own typical gaps
            # without a new best, that bet is worth less than the same workers
            # spent on a block still descending -- so shrink K and let the
            # queue hand the capacity to whoever is still improving.
            #
            # K never affects which node is popped, so throttling changes the
            # schedule and not the search: T_K = T_1 still holds, and the
            # block resumes its full width the moment it improves again.
            # Yielding is only a gift if somebody can take it. At the end of a
            # circuit one straggler block runs alone, and then shrinking K hands
            # its workers to nobody: measured on rc_adder_6 msz=4, the final
            # 195 s of a 404 s run kept 7.6 of 112 workers busy. Use the
            # scarcity test unconditionally: it was 1.346x faster and
            # bit-identical on rc_adder_6 msz=4 (job 1023530). The flag was
            # deleted rather than defaulted because a speedup that does not
            # change the output has no business being optional. The test uses
            # the same measured count K is sized from -- a saturated machine
            # reports ~0 idle, which is below any width.
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
                        record_spec_metric('stall_throttled_rounds')
                    else:
                        # Counted, not silent: a suppression that leaves no
                        # trace is indistinguishable from a throttle that never
                        # fired, and telling those apart is the whole point of
                        # the A/B.
                        record_spec_metric('stall_yield_suppressed')

            # Poll at most once per round: RuntimeFuture._done warns that
            # busy-wait polling can deadlock the runtime task.
            # Poll each outstanding flight once per round. `_done` is checked
            # rather than awaited: the runtime has no wait-any across futures,
            # and awaiting one would block the search on speculation -- the
            # exact inversion this design forbids.
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
                            record_spec_metric('spec_stale_epoch')
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
                logical_pop_cost = (
                    frontier.topk_costs(1)[0] if _LTRACE_DIR else 0.0
                )
                top_circuit, top_layer = frontier.pop()
                if _LTRACE_DIR:
                    _logical_trace({
                        'event': 'pop',
                        'cost': float(logical_pop_cost),
                        'layer': top_layer,
                    })

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
                if prefetched_initial_future is not None:
                    # This is not speculation. The initial layer was the only
                    # node in the frontier, so its expansion was certain; the
                    # future was launched before initial-layer instantiation
                    # and is consumed here instead of dispatching the same
                    # (successor, start) pairs a second time.
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
                        record_spec_metric('spec_stale_epoch')
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
                        record_spec_metric('spec_eventual_hits')
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
                            record_spec_metric('spec_hits')
                            record_spec_metric('spec_timely_hits')
                            _spec_hits += 1
                        else:
                            node_results = None
                    elif in_flight_batch is not None:
                        record_spec_metric('spec_eventual_hits')
                        record_spec_metric('spec_late')
                        # Never await speculation for a logical pop. Reuse
                        # only its immutable templates and submit fresh
                        # critical instantiations below.
                        node_successors = in_flight_batch
                        node_results = None
                    else:
                        node_successors = list(
                            layer_gen.gen_successors(top_circuit, data),
                        )
                        node_successors = note_pre_dup(node_successors)
                        node_results = None
                    if node_results is None and node_successors:
                        record_spec_metric('spec_misses')
                    popped_expansions.append((node_successors, node_results))
                    successors.extend(node_successors)
                    successor_layers.extend(
                        [top_layer] * len(node_successors),
                    )
                    n_node_successors = len(node_successors)
                else:
                    n_node_successors = 0
                    _plain = note_pre_dup(
                        list(layer_gen.gen_successors(top_circuit, data)),
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

            if first_expansion_future is not None or effective_k < 2:
                tasks_dispatched += len(successors)
                # The K=1 path needs the same accounting as the K>=2 path.
                # Without it the BASELINE every measurement is compared
                # against records zero rounds, zero tasks and zero stall,
                # which makes "did dedup reduce work?" unanswerable -- the
                # counters existed only on the arm that did not need them.
                record_spec_metric('critical_tasks', len(successors))
                record_spec_metric('dispatch_rounds')
                if aggregate_enabled:
                    record_spec_metric('critical_payload_ops', sum(
                        c.num_operations for c in successors
                    ))

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
                    # Clear the queue ahead of critical work.
                    #
                    # Dispatch ORDER is not enough. The runtime is FIFO and
                    # non-preemptive -- no priority queue, no wait-any -- so
                    # speculation dispatched in an earlier round is already
                    # sitting in front of this batch, and critical work waits
                    # behind it. On a saturated machine that is the whole
                    # story: measured on a 20-qubit whole-circuit compile
                    # where stock already reached peak occupancy 0.998,
                    # runahead came out at 0.89x -- 12% SLOWER than stock --
                    # while busy barely moved (0.294 -> 0.313). Speculation
                    # was not filling idle capacity, it was displacing the
                    # critical path.
                    #
                    # Since the queue cannot be reordered, it is emptied
                    # instead. Anything already finished was harvested at the
                    # top of this round, so only partially-executed work is
                    # lost, and rolling refill re-issues speculation behind
                    # critical immediately afterwards. Cancelling cannot
                    # affect correctness: speculative results are advisory,
                    # and a cancelled one simply becomes a cache miss that is
                    # recomputed on the critical path if it is ever needed.
                    if _SPEC_YIELD and speculation_flights:
                        _cancelled = sum(
                            f.n_tasks for f in speculation_flights
                        )
                        for _flight in speculation_flights:
                            get_runtime().cancel(_flight.future)
                        speculation_flights = []
                        record_spec_metric('spec_cancelled', _cancelled)
                        record_spec_metric('spec_yield_rounds')
                    # Queue critical work first so workers cannot choose newly
                    # dispatched speculation ahead of the search path.
                    critical_future, critical_owners = dispatch_batches(
                        critical_batches,
                    )
                    _n = sum(len(batch) for batch in critical_batches)
                    tasks_dispatched += _n
                    record_spec_metric('critical_tasks', _n)
                    record_spec_metric('dispatch_rounds')
                    if aggregate_enabled:
                        record_spec_metric('critical_payload_ops', sum(
                            c.num_operations
                            for batch in critical_batches for c in batch
                        ))

                # Refill rather than wait for a batch boundary. The budget
                # reserves one critical batch (s tasks) at all times, so the
                # resource rule holds continuously instead of once per batch.
                _s = max(1, round(succ_ema)) if succ_ema else len(successors)
                _s = max(1, _s)
                _in_flight = sum(f.n_tasks for f in speculation_flights)
                _budget = max(0, (effective_k * _s) - _s - _in_flight)
                if _budget > 0:
                    queued_keys: set[_CircuitStructureKey] = set()
                    for _flight in speculation_flights:
                        queued_keys.update(_flight.keys)
                    next_keys: list[_CircuitStructureKey] = []
                    next_batches: list[list[Circuit]] = []
                    _acc = 0
                    # Peek PAST the cached prefix, not merely wider than K.
                    #
                    # peek() returns the best nodes, and the best nodes are
                    # exactly the ones speculated in earlier rounds, so they
                    # are skipped by the two conditions below. Only one node
                    # leaves the frontier per round, so a K-wide window is
                    # almost entirely stale: measured at K=32, 31 nodes were
                    # requested and 1.03 new ones per round were found, a 3.3%
                    # fill rate that fell as 1/(K-1) -- the signature of a
                    # window that never reaches past what is already cached.
                    #
                    # The window therefore has to clear the cached prefix
                    # before it starts finding work. Deeper nodes are less
                    # likely to be popped soon, so H_timely falls, but the
                    # cache lives for the whole synthesis and an unused
                    # speculation on an otherwise idle worker costs nothing
                    # but memory, which is not the scarce resource here.
                    _peek = len(speculation_memo) + effective_k * 2 + 8
                    for _, circuit, _ in frontier.peek(_peek):
                        if _acc >= _budget:
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
                        node_successors = note_pre_dup(node_successors)
                        if not node_successors:
                            continue
                        if _acc + len(node_successors) > _budget:
                            break
                        _acc += len(node_successors)
                        queued_keys.add(structure_key)
                        next_keys.append(structure_key)
                        next_batches.append(node_successors)

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
                            epoch, current_max_layer, _acc,
                        ))
                        tasks_dispatched += _acc
                        record_spec_metric('spec_tasks', _acc)
                        record_spec_metric('spec_flights')
                        if aggregate_enabled:
                            record_spec_metric('spec_payload_ops', sum(
                                c.num_operations
                                for batch in next_batches for c in batch
                            ))
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

            map_id = None
            t_map_start = None
            t_map_end = None

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
                record_spec_metric('critical_stall_s', _batch_s)
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
                    record_spec_metric('contention_sum', contention_ema)
                    record_spec_metric('contention_samples')

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
            elif leapwaste_enabled:
                _t_stall1 = time.perf_counter()
                t_map_start = time.time()
                map_future = get_runtime().map(
                    Circuit.instantiate,
                    successors,
                    target=utry,
                    cost_hints=[
                        float(4 ** circuit.num_qudits)
                        for circuit in successors
                    ],
                    **instantiate_options,
                )
                map_id = getattr(map_future, '_bqprof_leapwaste_map_id', None)
                circuits = await map_future
                t_map_end = time.time()
                record_spec_metric(
                    'critical_stall_s', time.perf_counter() - _t_stall1,
                )
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
                # tail is not expressible here.
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
                _t_stall2 = time.perf_counter()
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
                record_spec_metric(
                    'critical_stall_s', time.perf_counter() - _t_stall2,
                )

            # Evaluate successors
            for win_index, circuit in enumerate(circuits):
                # Depth of *this* successor's parent, not the round's first.
                layer = successor_layers[win_index]
                dist = self.cost.calc_cost(circuit, utry)

                if dist < self.success_threshold:
                    _logical_trace({
                        'event': 'success',
                        'layer': layer + 1,
                        'distance': float(dist),
                    })
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
                    finish_speculation()
                    if aggregate_enabled:
                        if n_rollbacks > 0:
                            _LEAPWASTE_AGG_STATE['n_rollback_rescued'] += 1
                        _leapwaste_aggregate_record(
                            frontier_len_before_pop,
                            len(successors),
                            layer,
                            prefix_formed,
                            True,
                            len(circuits) - win_index - 1,
                            n_cleared,
                        )
                        if deepen_raised:
                            _LEAPWASTE_AGG_STATE[
                                'deepen_solved_after_raise'
                            ] += 1
                        _leapwaste_aggregate_finish()
                    emit_blockprof('success', layer + 1, float(dist))
                    abandon_prefetched_initial()
                    return circuit

                if self.check_new_best(layer + 1, dist, best_layer, best_dist):
                    _logical_trace({
                        'event': 'new_best',
                        'layer': layer + 1,
                        'distance': float(dist),
                    })
                    if _BLOCKPROF_DIR:
                        _bp_traj.append(
                            (time.perf_counter() - _bp_start,
                             layer + 1, float(dist)),
                        )
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

                    if self.check_leap_condition(
                        layer + 1,
                        best_dist,
                        best_layers,
                        best_dists,
                        last_prefix_layer,
                    ):
                        _logger.debug(f'Prefix formed at {layer + 1} layers.')
                        _logical_trace({
                            'event': 'prefix', 'layer': layer + 1,
                        })
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

                        # Commit stores the value from BEFORE this prefix, so
                        # a rollback lands at the branch point rather than at
                        # the prefix that turned out to be wrong. Advance only
                        # after it has been stored -- the order is the whole
                        # correctness of the rollback.
                        #
                        # `regret` rides along so a later rollback can choose
                        # WHICH commit to return to instead of always the most
                        # recent. It is how much cheaper the best candidate
                        # being discarded was than the one being kept: P0-d
                        # measured that 85-93% of prefix formations throw away
                        # something the frontier itself ranks cheaper, median
                        # gap 0.083-0.114. A large gap means the decision was
                        # forced against the frontier's own ordering, which
                        # makes it the likeliest mistake to undo.
                        _remaining = frontier.topk_costs(1)
                        _regret = (
                            frontier.score(circuit) - _remaining[0]
                            if _remaining else 0.0
                        )
                        # `layer` is stored so a rollback can price the
                        # commit by what it COST -- the depth burned after it
                        # -- rather than only by how arbitrary it looked at
                        # the time. Conflict-directed backjumping wants the
                        # decision the conflict depends on, and here the
                        # conflict is depth exhaustion, so the decision that
                        # burned the most budget is the candidate `regret`
                        # does not measure.
                        frontier.commit({
                            'last_prefix_layer': last_prefix_layer,
                            'regret': _regret,
                            'layer': layer + 1,
                        })
                        _logical_trace({
                            'event': 'commit',
                            'depth': frontier.committed_depth(),
                        })
                        last_prefix_layer = layer + 1
                        if add_child(circuit, layer + 1):
                            if leapwaste_enabled:
                                n_added_this_iter += 1

                if self.store_partial_solutions:
                    if layer not in psols:
                        psols[layer] = []

                    psols[layer].append((circuit.copy(), dist))

                    if len(psols[layer]) > self.partials_per_depth:
                        psols[layer].sort(key=lambda x: x[1])
                        del psols[layer][-1]

                if add_child(circuit, layer + 1):
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

        finish_speculation()
        if aggregate_enabled:
            _LEAPWASTE_AGG_STATE['n_exhausted_unverified'] += 1
            _leapwaste_aggregate_finish()
        # The exit that matters for a timeout: the search ran out of room
        # rather than finding a verified answer, so the trajectory recorded
        # here is the evidence for whether more budget would have helped.
        emit_blockprof('exhausted', best_layer, float(best_dist))
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
        """How much of this node is free right now, in cores, or None.

        Free CORES, not unassigned workers. Sizing speculation from the worker
        count under-committed the machine threefold -- LEAP saw 15.5 idle of
        112 while 41% of the cores were not computing -- and that is what
        pinned occupancy at 59%.

        The manager broadcasts (idle, total) into the worker cache every
        ``_OCCUPANCY_INTERVAL`` seconds. None means the reading is unavailable
        or stale, and callers must then fall back to whatever they did before
        -- None is "unknown", never "nothing is free". Reading absent-as-zero
        would disable speculation in every attached run, where no manager
        exists to broadcast at all.

        Staleness matters more than precision here: a reading from several
        seconds ago describes a machine that has since emptied or filled, and
        acting on it is worse than acting on the estimator, which is at least
        derived from this block's own current behaviour.
        """
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
