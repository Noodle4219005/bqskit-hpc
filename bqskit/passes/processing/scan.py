"""This module implements the ScanningGateRemovalPass."""
from __future__ import annotations

import logging
import os  # HPC
import time  # HPC
from typing import Any
from typing import Callable

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.operation import Operation
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.runtime import get_runtime  # HPC
from bqskit.utils.typing import is_real_number
_logger = logging.getLogger(__name__)

# ==================== HPC: scan lookahead ==========================================
# In auto mode, use the manager's free-core broadcast for each window.
# The cap follows acceptance rate rather than machine capacity; zero restores
# the original sequential loop.
_SCAN_LOOKAHEAD_TEXT = os.environ.get('BQSKIT_SCAN_LOOKAHEAD', 'auto')
_SCAN_LOOKAHEAD_AUTO = _SCAN_LOOKAHEAD_TEXT.strip().lower() == 'auto'
_SCAN_LOOKAHEAD = 8 if _SCAN_LOOKAHEAD_AUTO else int(_SCAN_LOOKAHEAD_TEXT)

if _SCAN_LOOKAHEAD < 0:
    raise ValueError('BQSKIT_SCAN_LOOKAHEAD must be non-negative or "auto".')

_SCAN_OCCUPANCY_STALE_AFTER = 1.0
_SCAN_LOOKAHEAD_CAP = int(os.environ.get('BQSKIT_SCAN_LOOKAHEAD_CAP', '32'))


def _scan_free_cores() -> int | None:
    """Return a fresh free-core reading, or None when unavailable."""
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
    """Try one removal without changing the supplied baseline."""
    wc = circuit.copy()
    wc.pop((cycle, qudit))
    wc.instantiate(target, **kwargs)
    return wc
# ===================================================================================


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
        # ==================== HPC: scan lookahead ==================================
        if _SCAN_LOOKAHEAD > 0:
            candidates: list[tuple[int, Operation]] = []
            for cycle, op in circuit.operations_with_cycles(
                reverse=reverse_iter,
            ):
                if not self.collection_filter(op):
                    _logger.debug(
                        f'Skipping operation {op} at cycle {cycle}.',
                    )
                    continue
                candidates.append((cycle, op))

            next_candidate = 0
            while next_candidate < len(candidates):
                window_start = next_candidate
                _k = _SCAN_LOOKAHEAD
                if _SCAN_LOOKAHEAD_AUTO:
                    _free = _scan_free_cores()
                    if _free is not None:
                        _k = max(
                            _SCAN_LOOKAHEAD,
                            min(_free, _SCAN_LOOKAHEAD_CAP),
                        )
                window = candidates[next_candidate:next_candidate + _k]
                next_candidate += len(window)

                # A candidate's stored cycle belongs to the original circuit.
                # Calculate its shift once from the shared baseline so an
                # unaccepted speculative removal cannot affect later choices.
                idx_shift = 0
                if self.start_from_left:
                    idx_shift = circuit.num_cycles - circuit_copy.num_cycles
                shifted_cycles = [cycle - idx_shift for cycle, _ in window]
                qudits = [op.location[0] for _, op in window]

                working_copies: list[Circuit] = await get_runtime().map(
                    _try_removal,
                    [circuit_copy] * len(window),
                    [target] * len(window),
                    shifted_cycles,
                    qudits,
                    cost_hints=[
                        float(4 ** circuit_copy.num_qudits)
                    ] * len(window),
                    **instantiate_options,
                )

                for index, ((cycle, op), working_copy) in enumerate(
                    zip(window, working_copies),
                ):
                    _logger.debug(
                        f'Attempting removal of operation at cycle {cycle}.',
                    )
                    _logger.debug(f'Operation: {op}')

                    if self.cost(working_copy, target) < self.success_threshold:
                        _logger.debug('Successfully removed operation.')
                        circuit_copy = working_copy
                        # Rewind after acceptance so the discarded tail is
                        # retried rather than skipped; this preserves the
                        # sequential greedy result.
                        next_candidate = window_start + index + 1
                        break

            circuit.become(circuit_copy)
            return
        # ===========================================================================

        for cycle, op in circuit.operations_with_cycles(reverse=reverse_iter):

            if not self.collection_filter(op):
                _logger.debug(f'Skipping operation {op} at cycle {cycle}.')
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
            working_copy.instantiate(target, **instantiate_options)

            if self.cost(working_copy, target) < self.success_threshold:
                _logger.debug('Successfully removed operation.')
                circuit_copy = working_copy

        circuit.become(circuit_copy)


def default_collection_filter(op: Operation) -> bool:
    return True
