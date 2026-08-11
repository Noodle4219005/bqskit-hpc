from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cpi import CPIGate
from bqskit.ir.gates.parameterized.u8 import U8Gate
from bqskit.passes import LEAPSynthesisPass
from bqskit.passes.search.generators.simple import SimpleLayerGenerator
from bqskit.qis import UnitaryMatrix


def test_leap_retry_increments_rollback_counter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    np.random.seed(0)
    monkeypatch.setenv('BQPROF_LEAPWASTE_AGGREGATE', '1')
    monkeypatch.setenv('BQPROF_LEAPWASTE_AGG_DIR', str(tmp_path))
    monkeypatch.setenv('BQSKIT_MAX_ROLLBACKS', '1')

    target = UnitaryMatrix.random(2, [3, 3])
    circuit = Circuit.from_unitary(target)
    leap = LEAPSynthesisPass(
        layer_generator=SimpleLayerGenerator(CPIGate(), U8Gate()),
        max_layer=3,
        min_prefix_size=1,
    )

    with Compiler(num_workers=1) as compiler:
        compiler.compile(circuit, [leap], data={'seed': 0})

    aggregate_paths = list(tmp_path.glob('leapagg_*.json'))
    assert len(aggregate_paths) == 1
    with aggregate_paths[0].open(encoding='utf-8') as handle:
        aggregate = json.load(handle)

    assert aggregate['n_rollbacks'] > 0
