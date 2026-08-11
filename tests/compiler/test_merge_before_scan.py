from __future__ import annotations

from bqskit.compiler.compile import build_gate_deletion_optimization_workflow
from bqskit.compiler.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import U3Gate
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.partitioning.single import GroupSingleQuditGatePass


def test_merge_before_scan_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv('BQSKIT_MERGE_BEFORE_SCAN', raising=False)

    workflow = build_gate_deletion_optimization_workflow()

    assert not any(
        isinstance(pass_obj, GroupSingleQuditGatePass)
        for pass_obj in workflow
    )


def test_merge_before_scan_uses_less_than_filter(monkeypatch) -> None:
    monkeypatch.setenv('BQSKIT_MERGE_BEFORE_SCAN', '1')

    workflow = build_gate_deletion_optimization_workflow()

    assert isinstance(workflow[1], GroupSingleQuditGatePass)
    assert isinstance(workflow[2], ForEachBlockPass)
    assert workflow[2].replace_filter == 'less-than'


def test_merge_before_scan_preserves_unitary_and_gate_count(
    monkeypatch,
    compiler: Compiler,
) -> None:
    monkeypatch.setenv('BQSKIT_MERGE_BEFORE_SCAN', '1')
    circuit = Circuit(1)
    for params in (
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5],
        [1.6, 1.7, 1.8],
    ):
        circuit.append_gate(U3Gate(), 0, params)

    input_unitary = circuit.get_unitary()
    input_num_operations = circuit.num_operations
    output_circuit = compiler.compile(
        circuit,
        build_gate_deletion_optimization_workflow(),
    )

    assert output_circuit.get_unitary().get_distance_from(input_unitary) < 1e-8
    assert output_circuit.num_operations <= input_num_operations
