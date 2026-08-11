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
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
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

            if _removed:
                _logger.debug('Successfully removed operation.')
                circuit_copy = working_copy

        if _probe_on:
            _probe['pass_seconds'] = round(time.perf_counter() - _t_pass, 6)
            for _key in ('inst_seconds_1q', 'inst_seconds_multi'):
                _probe[_key] = round(_probe[_key], 6)
            _scan_probe_emit(_probe)

        circuit.become(circuit_copy)


def default_collection_filter(op: Operation) -> bool:
    return True
