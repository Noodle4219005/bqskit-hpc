from __future__ import annotations

import pytest

from bqskit.passes import LEAPSynthesisPass


def test_deepen_schedule_is_parsed_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('BQSKIT_DEEPEN_SCHEDULE', 'double')
    monkeypatch.setenv('BQSKIT_DEEPEN_MAX', '4')

    leap = LEAPSynthesisPass(max_layer=4)

    assert leap.deepen_schedule == 'double'
    assert leap.deepen_budgets == (4, 8, 16, 32)


def test_luby_requires_random_tie_break_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('BQSKIT_DEEPEN_SCHEDULE', 'luby')
    monkeypatch.delenv('BQSKIT_RANDOM_TIEBREAK', raising=False)

    with pytest.raises(
        ValueError,
        match='BQSKIT_DEEPEN_SCHEDULE.*BQSKIT_RANDOM_TIEBREAK',
    ):
        LEAPSynthesisPass(max_layer=4)


def test_unset_schedule_preserves_max_layer_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('BQSKIT_DEEPEN_SCHEDULE', raising=False)
    monkeypatch.delenv('BQSKIT_DEEPEN_MAX', raising=False)

    bounded = LEAPSynthesisPass(max_layer=4)
    unbounded = LEAPSynthesisPass()

    assert bounded.max_layer == 4
    assert bounded.deepen_budgets == (4,)
    assert unbounded.max_layer is None
    assert unbounded.deepen_budgets == (None,)
