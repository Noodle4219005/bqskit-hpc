from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import bqskit.passes.synthesis.leap as leap_module
from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cpi import CPIGate
from bqskit.ir.gates.parameterized.u8 import U8Gate
from bqskit.passes import LEAPSynthesisPass
from bqskit.passes.search.generators.simple import SimpleLayerGenerator
from bqskit.qis import UnitaryMatrix


def test_deepen_to_unset_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('BQSKIT_DEEPEN_TO', raising=False)

    bounded = LEAPSynthesisPass(max_layer=4)
    unbounded = LEAPSynthesisPass()

    assert bounded.max_layer == 4
    assert bounded.deepen_to is None
    assert unbounded.max_layer is None
    assert unbounded.deepen_to is None


def test_deepen_to_without_max_layer_is_inert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('BQSKIT_DEEPEN_TO', '16')

    leap = LEAPSynthesisPass()

    assert leap.max_layer is None
    assert leap.deepen_to == 16


def test_deepen_to_restores_truncated_frontier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('BQSKIT_DEEPEN_TO', '5')
    monkeypatch.setenv('BQSKIT_MAX_ROLLBACKS', '0')
    monkeypatch.setenv('BQPROF_LEAPWASTE_AGGREGATE', '1')
    monkeypatch.setenv('BQPROF_LEAPWASTE_AGG_DIR', str(tmp_path))
    monkeypatch.setattr(leap_module, '_LEAPWASTE_DIR', str(tmp_path))
    monkeypatch.setattr(leap_module, '_LEAPWASTE_AGGREGATE', True)
    leap_module._LEAPWASTE_AGG_STATE.clear()

    np.random.seed(0)
    target = UnitaryMatrix.random(2, [3, 3])
    circuit = Circuit.from_unitary(target)
    leap = LEAPSynthesisPass(
        layer_generator=SimpleLayerGenerator(CPIGate(), U8Gate()),
        success_threshold=-1.0,
        max_layer=2,
        min_prefix_size=99,
    )

    with Compiler(num_workers=1) as compiler:
        compiler.compile(circuit, [leap], data={'seed': 0})

    aggregate_path = next(tmp_path.glob('leapagg_*.json'))
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))

    assert aggregate['deepen_overflow_max'] > 0
    assert aggregate['deepen_raises'] == 2
    assert aggregate['layer_hist'] == {
        '0': 1,
        '1': 1,
        '2': 1,
        '3': 1,
        '4': 1,
    }
