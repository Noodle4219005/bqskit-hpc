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
# consuming the answers in the original greedy order. Measured 2026-08-14 on
# square_heisenberg_N16 at msz=4 with 96 workers (job 1025050), against the
# sequential loop this replaces:
#
#   K    2Q  depth     wall     core-s  speedup  core-s delta
#   0    72   178   1191.7 s     13,211   1.00x        --
#   8    72   178    559.4 s     14,492   2.13x     +9.7%
#   32   72   178    504.2 s     18,832   2.36x    +42.5%
#
# Output is BIT-IDENTICAL at every K, which is what the cursor rollback buys: a
# window whose baseline moved is discarded and RETRIED, never skipped.
#
# 8 rather than 32 because K=32 buys 11% more wall for 4.4x the wasted work, and
# that waste is real core-seconds displacing other jobs. The ratio is the
# mechanism's own prediction -- waste = acceptance x K, single-qudit acceptance
# is 12.8% -- so it should carry to other circuits.
#
# 0 restores the original sequential loop verbatim.
_SCAN_LOOKAHEAD = int(os.environ.get('BQSKIT_SCAN_LOOKAHEAD', '8'))

if _SCAN_LOOKAHEAD < 0:
    raise ValueError('BQSKIT_SCAN_LOOKAHEAD must be non-negative.')


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
            next_candidate = 0
            while next_candidate < len(candidates):
                window_start = next_candidate
                window = candidates[
                    next_candidate:next_candidate + _SCAN_LOOKAHEAD
                ]
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

                _t_window = time.perf_counter() if _probe_on else 0.0
                working_copies: list[Circuit] = await get_runtime().map(
                    _try_removal,
                    [circuit_copy] * len(window),
                    [target] * len(window),
                    shifted_cycles,
                    qudits,
                    **instantiate_options,
                )
                _window_elapsed = (
                    time.perf_counter() - _t_window if _probe_on else 0.0
                )

                for index, ((cycle, op), working_copy) in enumerate(
                    zip(window, working_copies),
                ):
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
                        # Runtime.map exposes one elapsed time for the whole
                        # window, not one worker time per candidate. Splitting
                        # it preserves the pass-wall total while the two
                        # speculation counters record which candidate work
                        # was actually published and which was thrown away.
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
                        # All remaining answers were instantiated from the
                        # old circuit. A removal shrinks the reachable set, so
                        # even disjoint-looking gates cannot be safely reused:
                        # two individually removable RZ gates can fail when
                        # removed together. Start the next window afresh.
                        break

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
