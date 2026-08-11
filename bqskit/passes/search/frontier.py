"""This module implements the Frontier class."""
from __future__ import annotations

import heapq
import itertools
import os
from typing import Any
from typing import NamedTuple

from bqskit.ir.circuit import Circuit
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer

_GDFS = os.environ.get('BQPROF_GDFS') == '1'


class FrontierElement(NamedTuple):
    """The Frontier contains FrontierElements."""

    cost: float
    element_id: int
    circuit: Circuit
    extra_data: Any
    parent_id: int | None = None


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
        self._committed: list[tuple[list[FrontierElement], Any]] = []
        self._max_committed: int = int(
            os.environ.get('BQSKIT_MAX_COMMITTED', '8'),
        )
        self._counter = itertools.count()
        self._last_popped_id: int | None = None
        """Identity of the most recent pop, for the speculation probe."""
        self._last_popped_parent_id: int | None = None
        """Parent identity of the most recent pop, for the G-DFS probe."""

    def add(self, circuit: Circuit, extra_data: Any = None) -> None:
        """Add `circuit` into the frontier."""
        heuristic_value = self.heuristic_function(circuit, self.target)
        count = next(self._counter)
        parent_id = self._last_popped_id if _GDFS else None
        elem = FrontierElement(
            heuristic_value, count, circuit, extra_data, parent_id,
        )
        heapq.heappush(self._frontier, elem)

    def pop(self) -> tuple[Circuit, Any]:
        """Pop the top circuit."""
        elem = heapq.heappop(self._frontier)
        self._last_popped_id = elem.element_id
        if _GDFS:
            self._last_popped_parent_id = elem.parent_id
        return elem.circuit, elem.extra_data

    def __len__(self) -> int:
        """Return the number of nodes currently queued."""
        return len(self._frontier)

    def topk_ids(self, k: int) -> list[int]:
        """Return the element ids of the k cheapest entries, in order.

        Read-only: `heapq.nsmallest` does not disturb the heap. Used by the
        speculation probe to record what a speculative expansion would have
        dispatched, without dispatching anything.
        """
        if k <= 0 or not self._frontier:
            return []
        return [e.element_id for e in heapq.nsmallest(k, self._frontier)]

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

    def empty(self) -> bool:
        """Return true if the frontier is empty."""
        return len(self._frontier) == 0

    def clear(self) -> None:
        """Remove all elements from the frontier."""
        self._frontier.clear()

    def commit(self, state: Any = None) -> int:
        """
        Set the current frontier aside rather than destroying it.

        LEAP's prefix commit was `clear()`, which is irreversible: once the
        frontier is gone there is no way back to the branch point if the
        committed path turns out to be wrong. Measured, that matters -- when
        the search does climb back up the tree it climbs one level (p90 1, or
        2 on a real device graph), so recovery is cheap if it is possible at
        all.

        Setting the list aside rather than tagging its elements is deliberate.
        A per-element epoch would force `empty`, `__len__`, `prune`,
        `topk_ids`, `topk_costs` and `score` to all learn to skip stale
        entries, turning every read into a scan. Swapping the container leaves
        each of them looking at exactly the live frontier, so equivalence with
        `clear()` holds by construction rather than by argument.

        Args:
            state (Any): Optional caller state to restore with the frontier.

        Returns:
            int: The number of elements set aside.
        """
        count = len(self._frontier)
        self._committed.append((self._frontier, state))
        self._frontier = []
        # A cap is required, not tidiness. At min_prefix_size=3 one synthesis
        # forms up to 712 prefixes, and holding every frontier of ~120
        # circuits would be real memory. Past the cap the oldest is dropped
        # and becomes exactly as unrecoverable as it was before this change.
        while len(self._committed) > self._max_committed:
            self._committed.pop(0)
        return count

    def rollback(self) -> tuple[int, Any]:
        """
        Restore the most recently committed frontier, merging it back in.

        The state returned with the frontier lets callers restore any
        non-monotone state that belongs to the committed branch. In LEAP that
        is `last_prefix_layer` and nothing else: `best_circ`, `best_dist`,
        `best_layer` and `psols` all mean "best seen so far" and must survive a
        rollback.

        Returns:
            tuple[int, Any]: The number of elements restored and the state
                saved with the frontier, or ``(0, None)`` if nothing was
                committed.
        """
        if not self._committed:
            return 0, None
        prior, state = self._committed.pop()
        self._frontier.extend(prior)
        heapq.heapify(self._frontier)
        return len(prior), state

    def committed_depth(self) -> int:
        """Return how many committed frontiers are still recoverable."""
        return len(self._committed)
