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
from bqskit.utils.typing import is_integer


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
        self._frontier: list[FrontierElement] = []
        self._counter = itertools.count()
        self._last_popped_id: int | None = None
        """Identity of the most recent pop, for the speculation probe."""

    def add(self, circuit: Circuit, extra_data: Any = None) -> None:
        """Add `circuit` into the frontier."""
        heuristic_value = self.heuristic_function(circuit, self.target)
        count = next(self._counter)
        elem = FrontierElement(heuristic_value, count, circuit, extra_data)
        heapq.heappush(self._frontier, elem)

    def pop(self) -> tuple[Circuit, Any]:
        """Pop the top circuit."""
        elem = heapq.heappop(self._frontier)
        self._last_popped_id = elem.element_id
        return elem.circuit, elem.extra_data

    def topk_ids(self, k: int) -> list[int]:
        """Return the element ids of the k cheapest entries, in order.

        Read-only: `heapq.nsmallest` does not disturb the heap. Used by the
        speculation probe to record what a speculative expansion would have
        dispatched, without dispatching anything.
        """
        if k <= 0 or not self._frontier:
            return []
        return [
            e.element_id for e in heapq.nsmallest(k, self._frontier)
        ]

    def topk_costs(self, k: int) -> list[float]:
        """Return the heuristic costs of the k cheapest entries, in order.

        Also read-only. Used by the prefix-diversity probe to record what
        LEAP is about to throw away when a formed prefix clears the
        frontier: if those costs sit within noise of the candidate being
        kept, the choice of prefix is close to a coin flip and keeping
        several is worth something.
        """
        if k <= 0 or not self._frontier:
            return []
        return [e.cost for e in heapq.nsmallest(k, self._frontier)]

    def score(self, circuit: Circuit) -> float:
        """Return the heuristic cost `circuit` would get in this frontier.

        Evaluated against the same target and heuristic the frontier sorts
        by, so the result is directly comparable with `topk_costs`. Does not
        insert anything.
        """
        return self.heuristic_function(circuit, self.target)

    def empty(self) -> bool:
        """Return true if the frontier is empty."""
        return len(self._frontier) == 0

    def __len__(self) -> int:
        """Return the number of nodes currently queued."""
        return len(self._frontier)

    def clear(self) -> None:
        """Remove all elements from the frontier."""
        self._frontier.clear()

    def prune(self, k: int | None) -> int:
        """
        Keep only the `k` best nodes, discarding the rest.

        This bounds the frontier directly, by width, rather than indirectly
        through LEAP's prefix condition. Returns the number of nodes
        discarded, so callers can record how much was pruned.

        Args:
            k (int | None): The number of nodes to keep. `None` is a no-op,
                preserving the unbounded behaviour.

        Raises:
            ValueError: If `k` is not positive.
        """
        if k is None:
            return 0

        if not is_integer(k):
            raise TypeError(f'Expected integer for k, got {type(k)}.')

        if k <= 0:
            raise ValueError(f'Expected positive k, got {k}.')

        if len(self._frontier) <= k:
            return 0

        discarded = len(self._frontier) - k
        # FrontierElement orders by (heuristic, counter), so nsmallest
        # selects exactly the k the heap would have popped first.
        self._frontier = heapq.nsmallest(k, self._frontier)
        heapq.heapify(self._frontier)
        return discarded
