from __future__ import annotations

import pytest

from bqskit.compiler.compile import build_gate_deletion_optimization_workflow
from bqskit.compiler.workflow import Workflow
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.processing.scan import ScanningGateRemovalPass
from bqskit.passes.processing.treescan import TreeScanningGateRemovalPass


def get_gate_scan(workflow: Workflow) -> ScanningGateRemovalPass:
    partitioning_workflow = workflow[1]
    assert isinstance(partitioning_workflow, Workflow)
    block_pass = partitioning_workflow[2]
    assert isinstance(block_pass, ForEachBlockPass)
    scan_pass = block_pass.workflow[0]
    assert isinstance(scan_pass, ScanningGateRemovalPass)
    return scan_pass


def test_tree_scan_depth_unset_uses_plain_scan(monkeypatch) -> None:
    monkeypatch.delenv('BQSKIT_TREE_SCAN_DEPTH', raising=False)

    scan_pass = get_gate_scan(build_gate_deletion_optimization_workflow())

    # TreeScanningGateRemovalPass subclasses ScanningGateRemovalPass, so an
    # isinstance check alone would not distinguish the two implementations.
    assert type(scan_pass) is ScanningGateRemovalPass


def test_tree_scan_depth_one_uses_plain_scan(monkeypatch) -> None:
    monkeypatch.setenv('BQSKIT_TREE_SCAN_DEPTH', '1')

    scan_pass = get_gate_scan(build_gate_deletion_optimization_workflow())

    assert type(scan_pass) is ScanningGateRemovalPass


def test_tree_scan_depth_four_uses_tree_scan(monkeypatch) -> None:
    monkeypatch.setenv('BQSKIT_TREE_SCAN_DEPTH', '4')

    scan_pass = get_gate_scan(build_gate_deletion_optimization_workflow())

    assert type(scan_pass) is TreeScanningGateRemovalPass
    assert scan_pass.tree_depth == 4


@pytest.mark.parametrize('value', ['not-an-integer', '-1'])
def test_tree_scan_depth_rejects_invalid_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv('BQSKIT_TREE_SCAN_DEPTH', value)

    with pytest.raises(ValueError, match='BQSKIT_TREE_SCAN_DEPTH'):
        build_gate_deletion_optimization_workflow()
