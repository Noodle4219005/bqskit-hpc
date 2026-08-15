"""This module implements the ScanningGateRemovalPass."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any
from typing import Callable

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.operation import Operation
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.runtime import get_runtime
from bqskit.runtime.task import PRIORITY_CRITICAL
from bqskit.runtime.task import PRIORITY_SPECULATIVE
from bqskit.utils.typing import is_real_number
_logger = logging.getLogger(__name__)

# Line II probe: what is this scan spending its instantiations on?
#
# The loop in `run` does ONE FULL CIRCUIT INSTANTIATE PER OPERATION,
# serially, and `default_collection_filter` accepts every operation --
# including every single-qudit gate. A synthesised 3-qubit block measured
# here is 14 CNOTs and 31 single-qudit gates, so roughly 69% of those
# instantiations are spent trying to delete gates that cannot change
# `output_two_q_gates`, the metric V1 and V4 are written in, and that are
# frequently virtual on superconducting hardware anyway.
#
# Two nearly-free interventions follow if that holds, and this probe decides
# between them: pass a collection_filter that skips single-qudit gates, or
# merge adjacent single-qudit gates before the scan so that both the loop
# length and each instantiation's parameter vector shrink.
#
# Recorded by arity and split into attempts and successes, because those are
# different questions: attempts say what filtering would SAVE, successes say
# what it would COST.
_SCAN_PROBE_DIR = os.environ.get('BQPROF_SCAN_DIR')
_SCAN_FH_STATE: dict[str, Any] = {'pid': None, 'fh': None}
# How many undecided candidates to dispatch against the current baseline before
# consuming the answers in the original greedy order.
#
# Measured 2026-08-14 on square_heisenberg_N16 at msz=4 with 96 workers (job
# 1025050). These are the GATE-DELETION PHASE's own figures, cut out of the
# 250 ms CPU sampler with the pass timeline -- not sacct, which is the whole run
# and lets LEAP's 10,400 core-seconds dilute this phase by a factor of four:
#
#   K   phase wall   core-s  mean cores   /112  speedup  core-s delta
#   0      1010 s     2,856         2.8   2.5%    1.00x        --
#   8       370 s     4,009        10.8   9.7%    2.73x     +40.4%
#   32      318 s     8,353        26.3  23.5%    3.18x    +192.4%
#
# LEAP itself is untouched across the three (core-s +1.0%, wall 0.95x), so the
# change really is isolated to this pass.
#
# Output is BIT-IDENTICAL at every K, which is what the cursor rollback buys: a
# window whose baseline moved is discarded and RETRIED, never skipped. The
# invariant -- compute out of order, publish in search order -- does not mention
# K, so a window that differs every round is equally safe.
#
# 'auto' sizes each window from the free cores the manager broadcasts, re-read
# once per window so a machine that empties or fills is tracked, then capped by
# _SCAN_LOOKAHEAD_CAP below. An integer pins the window. 0 restores the original
# sequential loop verbatim.
_SCAN_LOOKAHEAD_TEXT = os.environ.get('BQSKIT_SCAN_LOOKAHEAD', 'auto')
_SCAN_LOOKAHEAD_AUTO = _SCAN_LOOKAHEAD_TEXT.strip().lower() == 'auto'
_SCAN_LOOKAHEAD = 8 if _SCAN_LOOKAHEAD_AUTO else int(_SCAN_LOOKAHEAD_TEXT)

if _SCAN_LOOKAHEAD < 0:
    raise ValueError('BQSKIT_SCAN_LOOKAHEAD must be non-negative or "auto".')

# Same value and the same reason as leap.py:216 -- a reading older than this
# describes a machine that has since changed, and acting on it is worse than
# falling back to the constant.
_SCAN_OCCUPANCY_STALE_AFTER = 1.0

# The window is capped by the ACCEPTANCE RATE, not by the machine. Measured
# 2026-08-14 on square_heisenberg_N16 at msz=4, each arm against the K=8 arm of
# its OWN job so the comparison is in-job:
#
#   K     wall vs that job's K=8   discarded speculations
#   8              --                       1,138
#   32          -9.9%  (job 1025050)          n/a
#   ~65         -2.7%  (job 1025213)         8,019
#
# The curve turns over. Sizing K from free cores read 64.7 idle of 96 and set
# K=64.7, which bought 2.7% of wall for 7.0x the discards -- worse than a flat
# 32. Single-qudit removals are accepted 12.8% of the time, so the expected
# number of candidates before the next acceptance is 1/0.128 = 7.8. A window
# past that speculates BEYOND the acceptance point, and everything past it is
# discarded and retried. Free cores are therefore the wrong ceiling: they say
# how much can run at once, not how much is worth running.
#
# 32 rather than 8 because dispatch still overlaps the waste -- k32 was the
# fastest arm measured. The cap is where the curve was still improving.
# 4096: a fuse, not a policy. Reverted to 32 and back again on 2026-08-16;
# the flip-flop is recorded here because the criterion, not the number, is the
# thing worth keeping.
#
# The useful work in this pass is a CONSTANT. Every candidate is solved exactly
# once, 3,556 of them on square_heisenberg, and no window size changes that.
# Job 1026188's deletion phase:
#
#   arm         wall     core-s   occupancy   discarded   useful occ   USEFUL core-s
#   map32     297.4 s     9,665       29.0%       59.5%        11.7%          3,910
#   drain32   223.1 s     9,107       36.4%       57.2%        15.6%          3,899
#   drainfree 214.3 s    10,550       44.0%       63.5%        16.0%          3,850
#
# The last column is flat. So K buys nothing except how fast the fixed useful
# work gets through and how much waste is dispatched alongside it. Against an
# energy objective the cap belongs at 32 (9,107 vs 10,550 core-seconds).
# Against a PARALLELISM objective it belongs off: 44.0% occupancy against
# 36.4%, the best useful occupancy of the three, and the shortest wall.
#
# The project objective is parallelism, so the fuse stays open. What this does
# NOT do is raise the ceiling: 3,850 useful core-seconds on 112 cores is 34.6 s
# of perfectly packed work inside a 214 s phase. The remaining 6.2x is
# dependency -- every accepted removal forces a new baseline -- and no window
# size reaches it.
_SCAN_LOOKAHEAD_CAP = int(os.environ.get('BQSKIT_SCAN_LOOKAHEAD_CAP', '4096'))

# Consume the window in index order AS ANSWERS ARRIVE instead of awaiting the
# whole map. The consumption loop already stops at the first accepted removal,
# so awaiting every result made a block wait on the slowest of K when it only
# ever needed the first ~1/p of them, where p is the acceptance rate.
#
# That barrier is the entire reason K needed a cap. A cap is a constant fitted
# to one circuit's acceptance rate; without the barrier an unconsumed
# speculation costs only the cores it happened to occupy, which the runtime's
# service classes already ration in favour of critical work.
#
# Measured on job 1025050 (square_heisenberg_N16, msz=4): the window advances
# 11.81 candidates per round against a mean K of 64.7, and 69.3% of everything
# dispatched was thrown away. On the block that owned the phase (142 ops,
# 174-245 s) the advance was 3.84, so it waited on 64 answers to use 4.
_SCAN_DRAIN = os.environ.get('BQSKIT_SCAN_DRAIN', '1') != '0'


def _scan_free_cores() -> int | None:
    """Free CORES on this node right now, or None when unknown.

    Mirrors ``LEAPSynthesisPass._measured_idle_workers``. None means the
    reading is absent or stale -- an attached run has no manager broadcasting
    at all -- and the caller must then use the fixed window, never zero.
    Reading absent-as-zero would silently turn speculation off wherever the
    broadcast does not reach, which is exactly the opt-in-default trap this
    branch keeps falling into.
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
    if time.monotonic() - stamp > _SCAN_OCCUPANCY_STALE_AFTER:
        return None
    return int(free)


def _try_removal(
    circuit: Circuit,
    target: Any,
    cycle: int,
    qudit: int,
    **kwargs: Any,
) -> Circuit:
    """Try one gate removal without modifying the supplied baseline."""
    wc = circuit.copy()
    wc.pop((cycle, qudit))
    wc.instantiate(target, **kwargs)
    return wc


def _scan_probe_emit(record: dict[str, Any]) -> None:
    """Append one ScanningGateRemovalPass summary, fork-safe."""
    if not _SCAN_PROBE_DIR:
        return
    try:
        pid = os.getpid()
        if _SCAN_FH_STATE['pid'] != pid:
            # Close the handle inherited across the fork before replacing it.
            # The other probes in this codebase (_foreach_emit,
            # _leapwaste_emit) leak one descriptor per forked worker here;
            # there is no reason to copy that.
            stale = _SCAN_FH_STATE.get('fh')
            if stale is not None:
                try:
                    stale.close()
                except Exception:
                    pass
            os.makedirs(_SCAN_PROBE_DIR, exist_ok=True)
            _SCAN_FH_STATE['fh'] = open(
                os.path.join(_SCAN_PROBE_DIR, f'scan_{pid}.jsonl'),
                'a',
                buffering=1,
            )
            _SCAN_FH_STATE['pid'] = pid
        _SCAN_FH_STATE['fh'].write(json.dumps(record) + '\n')
    except Exception:
        pass


def _mergeable_single_qudit_runs(
    circuit: Circuit,
) -> tuple[int, int, dict[int, int]]:
    """
    Count runs of consecutive single-qudit operations on one qudit.

    Returns:
        tuple[int, int, dict[int, int]]: How many runs of length >= 2 exist,
            how many operations would disappear if each run collapsed to ONE
            gate, and the full run-length histogram.

            The second number is NOT an achievable gain and must not be read
            as one. A single-qubit unitary in a native gate set costs up to
            five operations (ZXZXZ), so a run only shrinks if it is LONGER
            than five. The histogram is what answers that; collapse-to-one is
            an upper bound nothing can reach.
    """
    per_qudit: dict[int, list[tuple[int, int]]] = {}
    for cycle, op in circuit.operations_with_cycles():
        for qudit in op.location:
            per_qudit.setdefault(qudit, []).append((cycle, op.num_qudits))

    runs = saved = 0
    lengths: dict[int, int] = {}

    def close(streak: int) -> None:
        nonlocal runs, saved
        if streak >= 1:
            lengths[streak] = lengths.get(streak, 0) + 1
        if streak >= 2:
            runs += 1
            saved += streak - 1

    for ops in per_qudit.values():
        streak = 0
        for _, num_qudits in sorted(ops):
            if num_qudits == 1:
                streak += 1
                continue
            close(streak)
            streak = 0
        close(streak)
    return runs, saved, lengths


class ScanningGateRemovalPass(BasePass):
    """
    The ScanningGateRemovalPass class.

    Starting from one side of the circuit, attempt to remove gates one-by-one.
    """

    def __init__(
        self,
        start_from_left: bool = True,
        success_threshold: float = 1e-8,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        instantiate_options: dict[str, Any] = {},
        collection_filter: Callable[[Operation], bool] | None = None,
    ) -> None:
        """
        Construct a ScanningGateRemovalPass.

        Args:
            start_from_left (bool): Determines where the scan starts
                attempting to remove gates from. If True, scan goes left
                to right, otherwise right to left. (Default: True)

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the hilbert schmidt cost function.
                (Default: 1e-8)

            cost (CostFunction | None): The cost function that determines
                successful removal of a gate.
                (Default: HilbertSchmidtResidualsGenerator())

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

            collection_filter (Callable[[Operation], bool] | None):
                A predicate that determines which operations should be
                attempted to be removed. Called with each operation
                in the circuit. If this returns true, this pass will
                attempt to remove that operation. Defaults to all
                operations.
        """

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

        if not isinstance(instantiate_options, dict):
            raise TypeError(
                'Expected dictionary for instantiate_options, got %s.'
                % type(instantiate_options),
            )

        # Opt-in, read at construction rather than module import: capturing
        # an env var at import is what silently invalidated 38 cells of a
        # sweep in this project once.
        if collection_filter is None and os.environ.get(
            'BQSKIT_SKIP_CONSTANT_SQ',
        ) == '1':
            collection_filter = skip_incompensable_collection_filter
        self.collection_filter = collection_filter or default_collection_filter

        if not callable(self.collection_filter):
            raise TypeError(
                'Expected callable method that maps Operations to booleans for'
                ' collection_filter, got %s.' % type(self.collection_filter),
            )

        self.start_from_left = start_from_left
        self.success_threshold = success_threshold
        self.cost = cost
        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
            'min_iters': 100,
            'cost_fn_gen': self.cost,
        }
        self.instantiate_options.update(instantiate_options)

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        instantiate_options = self.instantiate_options.copy()
        if 'seed' not in instantiate_options:
            instantiate_options['seed'] = data.seed

        start = 'left' if self.start_from_left else 'right'
        _logger.debug(f'Starting scanning gate removal on the {start}.')

        target = self.get_target(circuit, data)

        circuit_copy = circuit.copy()
        reverse_iter = not self.start_from_left
        _probe: dict[str, Any] = {
            'n_ops': circuit.num_operations,
            'n_ops_1q': 0, 'n_ops_multi': 0,
            'attempts_1q': 0, 'attempts_multi': 0,
            'removed_1q': 0, 'removed_multi': 0,
            'inst_seconds_1q': 0.0, 'inst_seconds_multi': 0.0,
            'skipped_by_filter': 0,
        }
        _probe_on = bool(_SCAN_PROBE_DIR)
        if _probe_on:
            # The II-1 gate. docs/04 6 proposes merging adjacent single-qubit
            # runs before this loop, on the grounds that 69% of a block's
            # operations are single-qubit and they eat 82-85% of the
            # instantiate time. That only pays if consecutive 1Q gates on the
            # SAME qubit actually occur -- and LEAP emits CNOT(a,b) then
            # U3(a), U3(b), which puts a two-qudit gate between every
            # consecutive pair. Whether the circuit reaching this pass still
            # has that shape is a measurement, not an argument: the fully
            # compiled output has plenty to merge, but only because the rebase
            # to {CZ, RZ, SX, X} expands each U3 into RZ-SX-RZ-SX-RZ, and that
            # happens after this pass.
            _runs, _saved, _lengths = _mergeable_single_qudit_runs(circuit)
            _probe['mergeable_runs'] = _runs
            _probe['mergeable_ops_saved'] = _saved
            _probe['run_length_hist'] = {str(k): v for k, v in _lengths.items()}
            # What a merge can ACTUALLY remove: only the operations a run
            # carries beyond the five of a canonical ZXZXZ.
            _probe['ops_removable_vs_zxzxz'] = sum(
                count * (length - 5)
                for length, count in _lengths.items() if length > 5
            )
        _t_pass = time.perf_counter()
        if _SCAN_LOOKAHEAD > 0:
            candidates: list[tuple[int, Operation]] = []
            for cycle, op in circuit.operations_with_cycles(
                reverse=reverse_iter,
            ):
                if _probe_on:
                    if op.num_qudits >= 2:
                        _probe['n_ops_multi'] += 1
                    else:
                        _probe['n_ops_1q'] += 1

                if not self.collection_filter(op):
                    _logger.debug(
                        f'Skipping operation {op} at cycle {cycle}.',
                    )
                    if _probe_on:
                        _probe['skipped_by_filter'] += 1
                    continue

                candidates.append((cycle, op))

            # LEAP keeps 61.4 of 96 workers occupied because its independent
            # instantiations fan out through the runtime. This scan measured
            # 1,014 seconds but only 2,867 core-seconds, or 2.0 of 96 workers;
            # its 87--99.96% rejected removal attempts do not change the
            # baseline, so their answers can be made concurrently. Keep both
            # counts explicit: a low discarded count is the evidence that this
            # speculative work is buying occupancy rather than needlessly
            # re-instantiating an already obsolete circuit.
            _probe['speculations_used'] = 0
            _probe['speculations_discarded'] = 0
            # Declared before the loop, unconditionally, because an undeclared
            # key is not a missing statistic -- it is a KeyError that propagates
            # out of the pass and kills the compile. That is how all four arms
            # of job 1024323 died after 2:17:20 while exiting COMPLETED 0:0.
            _probe['window_from_measured'] = 0
            _probe['window_from_fixed'] = 0
            _probe['free_cores_seen_sum'] = 0
            _probe['window_size_sum'] = 0
            _probe['windows_cancelled'] = 0
            _probe['tasks_cancelled'] = 0
            _probe['accept_index_sum'] = 0
            _probe['windows_with_accept'] = 0
            _probe['window_latency_sum'] = 0.0
            next_candidate = 0
            while next_candidate < len(candidates):
                window_start = next_candidate
                # Window size is free to change between rounds without moving
                # the output. The invariant is "compute out of order, publish
                # in search order": answers are consumed in the original greedy
                # order and an accepted removal rolls the cursor back so the
                # rest of the window is RETRIED, not skipped. That holds for any
                # K, including a K that differs every round.
                _k = _SCAN_LOOKAHEAD
                if _SCAN_LOOKAHEAD_AUTO:
                    _free = _scan_free_cores()
                    if _free is None:
                        _probe['window_from_fixed'] += 1
                    else:
                        # max, not the reading alone: several blocks scan
                        # concurrently under ForEachBlockPass, so a node-wide
                        # reading is not this block's private share -- but the
                        # measured ceiling is the window, not the machine.
                        # Even K=32 left this phase at 23.5% of 112 cores
                        # (job 1025050), so erring high is what fills it.
                        _k = max(_SCAN_LOOKAHEAD, min(_free, _SCAN_LOOKAHEAD_CAP))
                        _probe['window_from_measured'] += 1
                        _probe['free_cores_seen_sum'] += _free
                        _probe['window_size_sum'] += _k
                window = candidates[next_candidate:next_candidate + _k]
                next_candidate += len(window)

                # A candidate's stored cycle belongs to the original circuit.
                # Every result in this window shares this one baseline, so the
                # left-to-right shift is calculated ONCE from that baseline.
                # Recomputing it from a speculative result would incorrectly
                # make an unaccepted removal affect a later decision.
                idx_shift = 0
                if self.start_from_left:
                    idx_shift = circuit.num_cycles - circuit_copy.num_cycles
                shifted_cycles = [cycle - idx_shift for cycle, _ in window]
                qudits = [op.location[0] for _, op in window]

                _t_window = time.perf_counter()
                _t_prev = _t_window
                # Head and tail go out at DIFFERENT service classes.
                #
                # Candidate 0 is the only one certain to be consumed: candidate
                # i is read iff no j < i was accepted, so P(read i) = (1-p)^i
                # and the measured 1/p here is 4.7. Everything after the head
                # is a bet.
                #
                # This pass never set task_priority at all, so Runtime.map's
                # default put the whole window -- up to the cap, and the cap is
                # now a fuse -- into PRIORITY_CRITICAL, the same service class
                # as every other block's head. With 16 blocks scanning at once
                # that is a head-of-line inversion: block A's candidate 0 queues
                # behind block B's candidate 30, which B will almost certainly
                # throw away. LEAP has passed task_priority through since
                # speculation was added; the scan was the one caller that did
                # not.
                #
                # Two maps rather than one because task_priority applies to a
                # whole batch. Both are dispatched before either is awaited, so
                # this costs no extra round trip.
                _fut_head = get_runtime().map(
                    _try_removal,
                    [circuit_copy], [target],
                    shifted_cycles[:1], qudits[:1],
                    task_priority=PRIORITY_CRITICAL,
                    cost_hints=[float(4 ** circuit_copy.num_qudits)],
                    **instantiate_options,
                )
                _n_tail = len(window) - 1
                _fut_tail = get_runtime().map(
                    _try_removal,
                    [circuit_copy] * _n_tail,
                    [target] * _n_tail,
                    shifted_cycles[1:],
                    qudits[1:],
                    task_priority=PRIORITY_SPECULATIVE,
                    cost_hints=[
                        float(4 ** circuit_copy.num_qudits)
                    ] * _n_tail,
                    **instantiate_options,
                ) if _n_tail > 0 else None
                # Index order of CONSUMPTION is what the output depends on, and
                # it is unchanged below. Only the waiting differs: _SCAN_DRAIN
                # pulls answers as they land, the old path blocks until all K
                # are in.
                _arrived: dict[int, Circuit] = {}
                _tail_out = _n_tail
                _outstanding = len(window)
                if not _SCAN_DRAIN:
                    _arrived = {0: (await _fut_head)[0]}
                    if _fut_tail is not None:
                        for _i, _r in enumerate(await _fut_tail):
                            _arrived[_i + 1] = _r
                    _tail_out = 0
                    _outstanding = 0
                _window_elapsed = (
                    time.perf_counter() - _t_window if _probe_on else 0.0
                )

                for index in range(len(window)):
                    cycle, op = window[index]
                    # Pull from whichever future still owes this index. The
                    # head map holds index 0 and nothing else, so there is no
                    # ambiguity and no wait-any needed.
                    while index not in _arrived:
                        if index == 0:
                            for _i, _r in await get_runtime().next(_fut_head):
                                _arrived[0] = _r
                                _outstanding -= 1
                        elif _tail_out > 0:
                            for _i, _r in await get_runtime().next(_fut_tail):
                                _arrived[_i + 1] = _r
                                _tail_out -= 1
                                _outstanding -= 1
                        else:
                            break
                    working_copy = _arrived.pop(index)
                    _logger.debug(
                        f'Attempting removal of operation at cycle {cycle}.',
                    )
                    _logger.debug(f'Operation: {op}')

                    _removed = (
                        self.cost(working_copy, target)
                        < self.success_threshold
                    )
                    _probe['speculations_used'] += 1
                    if _probe_on:
                        _arity = 'multi' if op.num_qudits >= 2 else '1q'
                        _probe[f'attempts_{_arity}'] += 1
                        # Draining gives a real per-candidate figure: the
                        # marginal wait since the previous candidate was
                        # consumed. It sums to the window latency instead of
                        # assuming a uniform split, which is what the old path
                        # had to do -- Runtime.map exposes one elapsed time for
                        # the whole window, not one worker time per candidate.
                        if _SCAN_DRAIN:
                            _t_now = time.perf_counter()
                            _inst_elapsed = _t_now - _t_prev
                            _t_prev = _t_now
                        else:
                            _inst_elapsed = _window_elapsed / len(window)
                        _probe[f'inst_seconds_{_arity}'] += _inst_elapsed
                        if _removed:
                            _probe[f'removed_{_arity}'] += 1
                        try:
                            _u = op.get_unitary()
                            _d = float(
                                _u.get_distance_from(
                                    UnitaryMatrix.identity(
                                        _u.dim, _u.radixes,
                                    ),
                                ),
                            )
                        except Exception:
                            _d = -1.0
                        if _d >= 0.0:
                            _b = min(int(_d * 20), 19)
                            _hist = _probe.setdefault(
                                'idist_'
                                f'{_arity}_'
                                f'{"removed" if _removed else "kept"}',
                                {},
                            )
                            _hist[str(_b)] = _hist.get(str(_b), 0) + 1
                        _by = _probe.setdefault(
                            f'gate_{"removed" if _removed else "kept"}',
                            {},
                        )
                        _name = type(op.gate).__name__
                        _by[_name] = _by.get(_name, 0) + 1
                        _sec = _probe.setdefault('gate_inst_seconds', {})
                        _sec[_name] = round(
                            _sec.get(_name, 0.0) + _inst_elapsed, 6,
                        )

                    if _removed:
                        _logger.debug('Successfully removed operation.')
                        circuit_copy = working_copy
                        _probe['speculations_discarded'] += (
                            len(window) - index - 1
                        )
                        # Roll the cursor back to just after the accepted
                        # candidate. Without this the discarded tail is SKIPPED
                        # rather than retried, and those gates never get their
                        # chance -- the output would then differ from
                        # BQSKIT_SCAN_LOOKAHEAD=0, which is the one property
                        # this whole change exists to preserve.
                        next_candidate = window_start + index + 1
                        _probe['accept_index_sum'] += index
                        _probe['windows_with_accept'] += 1
                        # All remaining answers were instantiated from the
                        # old circuit. A removal shrinks the reachable set, so
                        # even disjoint-looking gates cannot be safely reused:
                        # two individually removable RZ gates can fail when
                        # removed together. Start the next window afresh.
                        break

                # Whatever is still in flight belongs to a window that has
                # already been decided. Cancelling returns those cores now
                # rather than at the end of the phase, and it is the half of
                # the barrier fix that the runtime, not the pass, pays for.
                if _tail_out > 0 and _fut_tail is not None:
                    get_runtime().cancel(_fut_tail)
                    _probe['windows_cancelled'] += 1
                    _probe['tasks_cancelled'] += _outstanding
                _probe['window_latency_sum'] += (
                    time.perf_counter() - _t_window
                )

            if _probe_on:
                _probe['pass_seconds'] = round(
                    time.perf_counter() - _t_pass, 6,
                )
                for _key in ('inst_seconds_1q', 'inst_seconds_multi'):
                    _probe[_key] = round(_probe[_key], 6)
                _scan_probe_emit(_probe)

            circuit.become(circuit_copy)
            return

        for cycle, op in circuit.operations_with_cycles(reverse=reverse_iter):

            if _probe_on:
                if op.num_qudits >= 2:
                    _probe['n_ops_multi'] += 1
                else:
                    _probe['n_ops_1q'] += 1

            if not self.collection_filter(op):
                _logger.debug(f'Skipping operation {op} at cycle {cycle}.')
                if _probe_on:
                    _probe['skipped_by_filter'] += 1
                continue

            _logger.debug(f'Attempting removal of operation at cycle {cycle}.')
            _logger.debug(f'Operation: {op}')

            working_copy = circuit_copy.copy()

            # If removing gates from the left, we need to track index changes.
            if self.start_from_left:
                idx_shift = circuit.num_cycles
                idx_shift -= working_copy.num_cycles
                cycle -= idx_shift

            working_copy.pop((cycle, op.location[0]))
            # Guarded: these run once PER OPERATION, so with the probe off
            # they would be the only cost this instrumentation still charges.
            # The per-pass `_probe` dict and `_t_pass` above are once per
            # invocation and left unguarded for readability.
            _t_inst = time.perf_counter() if _probe_on else 0.0
            working_copy.instantiate(target, **instantiate_options)
            _inst_elapsed = (
                time.perf_counter() - _t_inst if _probe_on else 0.0
            )

            _removed = self.cost(working_copy, target) < self.success_threshold
            if _probe_on:
                _arity = 'multi' if op.num_qudits >= 2 else '1q'
                _probe[f'attempts_{_arity}'] += 1
                _probe[f'inst_seconds_{_arity}'] += _inst_elapsed
                if _removed:
                    _probe[f'removed_{_arity}'] += 1
                # Is removability cheaply predictable?
                #
                # The LEAP paper's dimensionality reduction removes up to 40%
                # of the U3 gates, and the implementation is this loop: try
                # each gate in turn, pay one full-circuit instantiate, keep
                # the removal if it worked. P0-g measured a 12.8% success rate
                # on single-qudit gates, so 87% of those instantiates buy
                # nothing.
                #
                # If a gate's distance from the identity predicted its
                # removability, the loop could be ordered by it -- or stopped
                # early -- and most of the removals would come for a fraction
                # of the instantiates. Bucket the outcome by that distance so
                # the separation, if any, is visible. The distance is a 2x2
                # (or d x d) norm, free next to an instantiate.
                try:
                    _u = op.get_unitary()
                    _d = float(
                        _u.get_distance_from(
                            UnitaryMatrix.identity(_u.dim, _u.radixes),
                        ),
                    )
                except Exception:
                    _d = -1.0
                if _d >= 0.0:
                    _b = min(int(_d * 20), 19)
                    _hist = _probe.setdefault(
                        f'idist_{_arity}_{"removed" if _removed else "kept"}',
                        {},
                    )
                    _hist[str(_b)] = _hist.get(str(_b), 0) + 1
                # By gate type as well, because one bucket held 87 attempts
                # and zero successes and guessing which gate that was is
                # exactly the kind of structural assumption that has been
                # wrong three times in this project.
                _by = _probe.setdefault(
                    f'gate_{"removed" if _removed else "kept"}', {},
                )
                _name = type(op.gate).__name__
                _by[_name] = _by.get(_name, 0) + 1
                _sec = _probe.setdefault('gate_inst_seconds', {})
                _sec[_name] = round(_sec.get(_name, 0.0) + _inst_elapsed, 6)

            if _removed:
                _logger.debug('Successfully removed operation.')
                circuit_copy = working_copy

        if _probe_on:
            _probe['pass_seconds'] = round(time.perf_counter() - _t_pass, 6)
            for _key in ('inst_seconds_1q', 'inst_seconds_multi'):
                _probe[_key] = round(_probe[_key], 6)
            _scan_probe_emit(_probe)

        circuit.become(circuit_copy)


def skip_incompensable_collection_filter(op: Operation) -> bool:
    """
    Skip gates whose removal the remaining free parameters cannot compensate.

    The LEAP paper's dimensionality reduction removes up to 40% of the U3
    gates, and this pass is the implementation: try each gate in turn, pay one
    full-circuit instantiate, keep the removal if the result still meets the
    threshold. Measured per gate type, that budget is badly spent:

        gate        attempts  removed  success  share of instantiate time
        RZGate            91       23    25.3%                      36.8%
        SqrtXGate         76        0     0.0%                      44.4%
        CZGate            27        3    11.1%                      18.8%

    CORRECTED 2026-08-12. On the three acceptance circuits SqrtX removal
    succeeded 0 times in 76, and this docstring claimed it never succeeds and
    that skipping it was free. On the P0-g circuits it succeeds 374 times in
    17,664 -- 2.1%, not 0 -- so 76 attempts was too small a sample and the
    structural argument below was drawn too strongly.

    The argument still explains WHY the rate is low. In the native set
    {RZ, SX, X} the RZ gates are diagonal, so a maximal single-qudit run is
    RZ-SX-RZ-SX-RZ with exactly two sources of off-diagonal structure. Drop
    one and what remains is D1 * SX * D2, a two-parameter family, while a
    general SU(2) element needs three. Removal can only succeed when the run
    happens to lie in that restricted family, which is rare -- but not never.

    So this is a PRICED TRADE, not a free win. Measured on ham15-med,
    ham15-low and adder_8 at 96 workers:

        gate        attempts  removed  success  share of instantiate time
        SqrtXGate     17,664      374     2.1%                     50.6%
        RZGate        15,182    3,888    25.6%                     29.6%
        CZGate         5,734       46     0.8%                     19.9%

        skip off   wall 261.0 s   2Q 1038   depth 2948   removed 4,308
        skip on    wall 209.6 s   2Q 1038   depth 3146   removed 3,604

    1.25x wall for +6.7% depth, 2Q unchanged. Total removals fall by more
    than SqrtX's own 374 because the scan is sequential and greedy, so
    skipping a gate changes the trajectory for everything after it.

    CZ is constant too and is deliberately NOT skipped, though at 0.8% its
    case is now weaker than the 11% first measured locally.

    Note this is the opposite trade to TreeScanningGateRemovalPass, which at
    tree_depth=2 buys 2.0% of depth for 4% of wall. The two knobs move along
    the same exchange rate in opposite directions.
    """
    return not (op.num_qudits == 1 and op.gate.num_params == 0)


def default_collection_filter(op: Operation) -> bool:
    return True
