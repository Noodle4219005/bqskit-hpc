"""This module implements the ZXZXZDecomposition."""
from __future__ import annotations

import cmath
import json
import os

from typing import Any

import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.sx import SqrtXGate
from bqskit.ir.gates.parameterized.rx import RXGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.gates.parameterized.u1 import U1Gate


# What does this decomposition actually emit? It appends RZ, SX, RZ, SX, RZ
# unconditionally, with no check on the three angles, so a pure Z rotation
# costs five gates. The gate-deletion scan then removes the redundant ones by
# brute force -- P0-g measured 1,993 successful single-qudit removals on
# ham15-med, each paid for with a full-circuit instantiate. This probe sizes
# what emitting the minimal form here instead would be worth.
_ZXZXZ_PROBE_DIR = os.environ.get('BQPROF_ZXZXZ_DIR')
_ZXZXZ_FH: dict[str, Any] = {'pid': None, 'fh': None}


def _zxzxz_emit(record: dict[str, Any]) -> None:
    """Append one decomposition record, fork-safe."""
    if not _ZXZXZ_PROBE_DIR:
        return
    try:
        pid = os.getpid()
        if _ZXZXZ_FH['pid'] != pid:
            stale = _ZXZXZ_FH.get('fh')
            if stale is not None:
                try:
                    stale.close()
                except Exception:
                    pass
            os.makedirs(_ZXZXZ_PROBE_DIR, exist_ok=True)
            _ZXZXZ_FH['fh'] = open(
                os.path.join(_ZXZXZ_PROBE_DIR, f'zxzxz_{pid}.jsonl'),
                'a', buffering=1,
            )
            _ZXZXZ_FH['pid'] = pid
        _ZXZXZ_FH['fh'].write(json.dumps(record) + '\n')
    except Exception:
        pass


class ZXZXZDecomposition(BasePass):
    """
    The ZXZXZDecomposition class.

    Convert a single-qubit circuit to ZXZXZ sequence.
    """

    def __init__(
        self,
        always_use_rx: bool = False,
        always_use_u1: bool = False,
    ) -> None:
        """
        Construct a ZXZXZDecomposition pass.

        Args:
            always_use_rx (bool): If True, always use RX instead of SX.

            always_use_u1 (bool): If True, always use U1 instead of RZ.
        """

        if not isinstance(always_use_rx, bool):
            raise TypeError(
                f'Expected bool for always_use_rx, got {type(always_use_rx)}.',
            )

        if not isinstance(always_use_u1, bool):
            raise TypeError(
                f'Expected bool for always_use_u1, got {type(always_use_u1)}.',
            )

        self.always_use_rx = always_use_rx
        self.always_use_u1 = always_use_u1

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        if circuit.num_qudits != 1:
            raise ValueError(
                'Cannot convert multi-qudit circuit into ZXZXZ sequence.',
            )

        if circuit.radixes[0] != 2:
            raise ValueError(
                'Cannot convert non-qubit circuit into ZXZXZ sequence.',
            )

        # Decide on RX or SX
        no_sx = RXGate() in data.gate_set and SqrtXGate() not in data.gate_set
        use_rx = self.always_use_rx or no_sx

        # Decide on RZ or U1
        no_rz = U1Gate() in data.gate_set and RZGate() not in data.gate_set
        use_u1 = self.always_use_u1 or no_rz

        utry = circuit.get_unitary()

        # Calculate params
        utry = np.linalg.det(utry) ** (-0.5) * utry
        i1 = cmath.phase(utry[1, 1])
        i2 = cmath.phase(utry[1, 0])
        t = 2 * np.arctan2(abs(utry[1, 0]), abs(utry[0, 0])) + np.pi
        p = i1 + i2 + np.pi
        l = i1 - i2

        # Move angles into [-pi, pi)
        t = (t + np.pi) % (2 * np.pi) - np.pi
        p = (p + np.pi) % (2 * np.pi) - np.pi
        l = (l + np.pi) % (2 * np.pi) - np.pi

        new_circuit = Circuit(1)

        if use_u1:
            new_circuit.append_gate(U1Gate(), 0, [l])
        else:
            new_circuit.append_gate(RZGate(), 0, [l])

        if use_rx:
            new_circuit.append_gate(RXGate(), 0, [np.pi / 2])
        else:
            new_circuit.append_gate(SqrtXGate(), 0)

        if use_u1:
            new_circuit.append_gate(U1Gate(), 0, [t])
        else:
            new_circuit.append_gate(RZGate(), 0, [t])

        if use_rx:
            new_circuit.append_gate(RXGate(), 0, [np.pi / 2])
        else:
            new_circuit.append_gate(SqrtXGate(), 0)

        if use_u1:
            new_circuit.append_gate(U1Gate(), 0, [p])
        else:
            new_circuit.append_gate(RZGate(), 0, [p])

        if _ZXZXZ_PROBE_DIR:
            def _triv(x: float) -> bool:
                y = float(x) % (2 * np.pi)
                return min(y, 2 * np.pi - y) < 1e-9
            _zxzxz_emit({
                'l': float(l), 't': float(t), 'p': float(p),
                'l_id': _triv(l), 't_id': _triv(t), 'p_id': _triv(p),
                't_pi': abs(abs(float(t)) - np.pi) < 1e-9,
                'in_ops': circuit.num_operations,
                'out_ops': new_circuit.num_operations,
            })

        circuit.become(new_circuit)
