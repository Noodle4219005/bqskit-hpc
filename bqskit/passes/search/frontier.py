"""This module implements the Frontier class."""
from __future__ import annotations

import heapq
import itertools
from typing import Any
from typing import NamedTuple

from bqskit.ir.circuit import Circuit
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

class FrontierElement(NamedTuple):
    """The Frontier contains FrontierElements."""

    cost: float
    element_id: int
    circuit: Circuit
    extra_data: Any


class Frontier:
    """The Frontier class."""

    def __init__(
        self,
        target: UnitaryMatrix | StateVector | StateSystem,
        heuristic_function: HeuristicFunction,
    ) -> None:
        """
        Construct an empty frontier.

        Args:
            target (UnitaryMatrix | StateVector | StateSystem): The target to
                pass to the heuristic_function.

            heuristic_function (HeuristicFunction): The heuristic used
                to sort the Frontier.
        """
        if not isinstance(target, (UnitaryMatrix, StateVector, StateSystem)):
            raise TypeError(
                'Expected unitary or state, got %s.' % type(target),
            )

        if not isinstance(heuristic_function, HeuristicFunction):
            raise TypeError(
                'Expected HeursiticFunction, got %s.'
                % type(heuristic_function),
            )

        self.target = target
        self.heuristic_function = heuristic_function
        # ==================== HPC: reversible commit ===============================
        self._frontier: list[FrontierElement] = []
        self._committed: list[tuple[list[FrontierElement], Any]] = []
        self._max_committed: int = int(
            os.environ.get('BQSKIT_MAX_COMMITTED', '8'),
        )
        # ===========================================================================
        self._counter = itertools.count()

    def add(self, circuit: Circuit, extra_data: Any = None) -> None:
        """Add `circuit` into the frontier."""
        heuristic_value = self.heuristic_function(circuit, self.target)
        count = next(self._counter)
        elem = FrontierElement(heuristic_value, count, circuit, extra_data)
        heapq.heappush(self._frontier, elem)

    def pop(self) -> tuple[Circuit, Any]:
        """Pop the top circuit."""
        elem = heapq.heappop(self._frontier)
        return elem.circuit, elem.extra_data

    def __len__(self) -> int:
        """Return the number of nodes currently queued."""
        return len(self._frontier)

    def topk_ids(self, k: int) -> list[int]:
        """Return the ids of the k cheapest entries without changing the heap."""
        if k <= 0 or not self._frontier:
            return []
        return [e.element_id for e in heapq.nsmallest(k, self._frontier)]

    def peek(self, k: int) -> list[tuple[int, Circuit, Any]]:
        """Return the k cheapest entries without changing the heap."""
        if k <= 0 or not self._frontier:
            return []
        return [
            (elem.element_id, elem.circuit, elem.extra_data)
            for elem in heapq.nsmallest(k, self._frontier)
        ]

    def topk_costs(self, k: int) -> list[float]:
        """Return the heuristic costs of the k cheapest entries."""
        if k <= 0 or not self._frontier:
            return []
        return [e.cost for e in heapq.nsmallest(k, self._frontier)]

    def score(self, circuit: Circuit) -> float:
        """Return the cost `circuit` would receive without adding it."""
        return self.heuristic_function(circuit, self.target)

    def empty(self) -> bool:
        """Return true if the frontier is empty."""
        return len(self._frontier) == 0

    def clear(self) -> None:
        """Remove all elements from the frontier."""
        self._frontier.clear()

    # ==================== HPC: reversible commit ===================================
    def commit(self, state: Any = None) -> int:
        """Set aside the live frontier and optionally save caller state."""
        count = len(self._frontier)
        self._committed.append((self._frontier, state))
        self._frontier = []
        # Keep a bounded history so recovery cannot retain unbounded frontiers.
        while len(self._committed) > self._max_committed:
            self._committed.pop(0)
        return count
    # ===============================================================================

    # ==================== HPC: rollback ============================================
    def rollback(self) -> tuple[int, Any]:
        """Restore the latest committed frontier and its saved state."""
        if not self._committed:
            return 0, None
        prior, state = self._committed.pop()
        self._frontier.extend(prior)
        heapq.heapify(self._frontier)
        return len(prior), state

    def committed_depth(self) -> int:
        """Return how many committed frontiers are still recoverable."""
        return len(self._committed)

    def committed_states(self) -> list[Any]:
        """Return retained states oldest first for targeted rollback."""
        return [state for _, state in self._committed]

    def rollback_to(self, index: int) -> tuple[int, Any]:
        """Restore a retained commit and discard its descendants."""
        if not self._committed:
            return 0, None

        if not 0 <= index < len(self._committed):
            raise IndexError(
                f'commit index {index} out of range '
                f'0..{len(self._committed) - 1}.',
            )

        # Descendant frontiers depend on the retracted decision and are invalid.
        del self._committed[index + 1:]
        return self.rollback()
    # ===============================================================================
