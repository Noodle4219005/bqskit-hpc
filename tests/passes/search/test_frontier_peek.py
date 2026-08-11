from __future__ import annotations

from bqskit.ir.circuit import Circuit
from bqskit.passes.search.frontier import Frontier
from bqskit.passes.search.heuristics import DijkstraHeuristic
from bqskit.qis import UnitaryMatrix


def test_peek_is_read_only_and_matches_pop_order() -> None:
    target = UnitaryMatrix.identity(1)
    frontier = Frontier(target, DijkstraHeuristic())
    for extra_data in range(3):
        frontier.add(Circuit(1), extra_data)

    peeked = frontier.peek(5)

    assert [entry[2] for entry in peeked] == [0, 1, 2]
    assert len(frontier) == 3
    assert frontier.peek(5) == peeked

    assert [frontier.pop()[1] for _ in range(3)] == [0, 1, 2]


def test_peek_returns_fewer_entries_than_requested() -> None:
    target = UnitaryMatrix.identity(1)
    frontier = Frontier(target, DijkstraHeuristic())
    frontier.add(Circuit(1), 'only-entry')

    assert len(frontier.peek(4)) == 1
