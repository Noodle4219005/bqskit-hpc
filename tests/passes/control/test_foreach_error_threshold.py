from __future__ import annotations

import pytest

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.passes.control.foreach import ForEachBlockPass


class NoOpPass(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        pass


def test_error_threshold_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('BQSKIT_BLOCK_ERROR_THRESHOLD', raising=False)

    foreach = ForEachBlockPass(NoOpPass())

    assert foreach.calculate_error_bound is False
    assert foreach.error_threshold is None


def test_error_threshold_forces_error_calculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('BQSKIT_BLOCK_ERROR_THRESHOLD', '1e-8')

    foreach = ForEachBlockPass(NoOpPass())

    assert foreach.calculate_error_bound is True
    assert foreach.error_threshold == 1e-8


def test_malformed_error_threshold_names_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('BQSKIT_BLOCK_ERROR_THRESHOLD', 'not-a-float')

    with pytest.raises(ValueError, match='BQSKIT_BLOCK_ERROR_THRESHOLD'):
        ForEachBlockPass(NoOpPass())
