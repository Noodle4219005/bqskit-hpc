from __future__ import annotations

import pytest

from bqskit.ir.circuit import Circuit
from bqskit.passes.search.frontier import Frontier
from bqskit.passes.search.heuristics import DijkstraHeuristic
from bqskit.qis import UnitaryMatrix


def test_commit_rollback_restores_frontier() -> None:
    target = UnitaryMatrix.identity(1)
    frontier = Frontier(target, DijkstraHeuristic())
    for extra_data in range(3):
        frontier.add(Circuit(1), extra_data)

    ids_before_commit = frontier.topk_ids(3)
    assert frontier.commit() == 3
    assert frontier.empty()
    assert frontier.committed_depth() == 1

    assert frontier.rollback() == (3, None)
    assert frontier.topk_ids(3) == ids_before_commit
    assert [frontier.pop()[1] for _ in range(3)] == [0, 1, 2]


def test_commit_cap_drops_oldest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('BQSKIT_MAX_COMMITTED', '2')
    target = UnitaryMatrix.identity(1)
    frontier = Frontier(target, DijkstraHeuristic())

    for extra_data in range(3):
        frontier.add(Circuit(1), extra_data)
        assert frontier.commit() == 1

    assert frontier.committed_depth() == 2
    assert frontier.rollback() == (1, None)
    assert frontier.pop()[1] == 2
    assert frontier.rollback() == (1, None)
    assert frontier.pop()[1] == 1
    assert frontier.rollback() == (0, None)


def test_commit_rollback_restores_state_with_frontier() -> None:
    target = UnitaryMatrix.identity(1)
    frontier = Frontier(target, DijkstraHeuristic())
    frontier.add(Circuit(1), 0)

    assert frontier.commit(state=7) == 1
    assert frontier.rollback() == (1, 7)
    assert frontier.pop()[1] == 0


def test_commit_cap_drops_oldest_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('BQSKIT_MAX_COMMITTED', '2')
    target = UnitaryMatrix.identity(1)
    frontier = Frontier(target, DijkstraHeuristic())

    for extra_data in range(3):
        frontier.add(Circuit(1), extra_data)
        assert frontier.commit(state=extra_data) == 1

    assert frontier.committed_depth() == 2
    assert frontier.rollback() == (1, 2)
    assert frontier.pop()[1] == 2
    assert frontier.rollback() == (1, 1)
    assert frontier.pop()[1] == 1
    assert frontier.rollback() == (0, None)
