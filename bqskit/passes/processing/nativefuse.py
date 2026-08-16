"""Exact algebraic fusion in the native {CZ, RZ, SX, X} gate set."""
from __future__ import annotations

import logging
import math
import os
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cz import CZGate
from bqskit.ir.gates.constant.sx import SqrtXGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.parameterized.rz import RZGate

_logger = logging.getLogger(__name__)

# ==================== HPC: native phase fusion =====================================
# The whole of this module. ZXZXZDecomposition expands every single-qudit run
# into RZ-SX-RZ-SX-RZ unconditionally, and gate deletion then runs after that
# expansion and pays a numerical re-instantiation per candidate to rediscover
# that some of those RZ are redundant. These identities settle the same
# question by inspection.
_TWO_PI = 2.0 * math.pi
_HALF_PI = math.pi / 2.0

# Strict on purpose. This pass does algebra, not approximate synthesis, so it
# must never be the reason a block drifts toward the success threshold.
# Anything looser is left for the numerical scan to judge.
_TOL = float(os.environ.get('BQSKIT_FUSE_TOL', '1e-12'))


class NativePhaseFusionPass(BasePass):
    """Fuse Z- and X-axis rotations exactly, in the native gate set.

    The identities, all exact equalities:

        RZ(a) RZ(b) = RZ(a+b)
        [CZ, RZ (x) I] = 0            both are diagonal
        RZ(2*pi) = -I                  a global phase
        SX SX = X
        X X = I
        CZ does NOT commute with SX    checked, and relied upon

    Dropping the 2*pi is sound because the distance this is judged by,
    UnitaryMatrix.get_distance_from, takes an absolute value of the trace and
    is therefore blind to global phase.

    This is not bit-identical to running without it. The circuit that reaches
    a later numerical pass is different, so that pass's seeded random starts
    differ and it may accept a different set of the remaining gates. The
    unitary is preserved exactly; the specific surviving gates are not.

    It finds only local redundancy: anything needing the whole block's
    parameters to be re-fitted is invisible to a rewrite rule.
    """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        # Alternate until nothing changes. The two axes feed each other: a
        # vanishing RZ leaves SX-SX adjacent, which becomes X, and two X
        # become nothing, which can leave two RZ adjacent again.
        while True:
            before = circuit.num_operations
            circuit.become(self._sweep(circuit))
            if circuit.num_operations >= before:
                break
        _logger.debug(
            'NativePhaseFusion: %d operations.', circuit.num_operations,
        )

    def _sweep(self, circuit: Circuit) -> Circuit:
        """One left-to-right pass, rebuilding the circuit."""
        out = Circuit(circuit.num_qudits, circuit.radixes)

        # Per qudit: the axis with an unemitted rotation, and its angle.
        # At most one axis can be pending at a time, because meeting a gate on
        # the other axis flushes this one first -- Z and X do not commute.
        axis: dict[int, str] = {}
        angle: dict[int, float] = {}
        def flush(q: int) -> None:
            a = axis.get(q)
            if a is None:
                return
            t = angle[q]
            axis.pop(q)
            angle.pop(q)
            if a == 'z':
                # Into (-pi, pi]. RZ(2*pi) is -I, i.e. global phase.
                t = (t + math.pi) % _TWO_PI - math.pi
                if abs(t) <= _TOL:
                    return
                out.append_gate(RZGate(), (q,), [t])
            else:
                # Units of pi/2 modulo 2*pi: 0 nothing, 1 SX, 2 X, 3 X then SX.
                k = int(round(t / _HALF_PI)) % 4
                if k == 0:
                    return
                if k >= 2:
                    out.append_gate(XGate(), (q,))
                if k % 2 == 1:
                    out.append_gate(SqrtXGate(), (q,))

        for op in circuit:
            gate = op.gate
            loc = [int(q) for q in op.location]
            if isinstance(gate, RZGate):
                q = loc[0]
                if axis.get(q) == 'x':
                    flush(q)
                if axis.get(q) == 'z':
                    angle[q] += float(op.params[0])
                else:
                    axis[q] = 'z'
                    angle[q] = float(op.params[0])
            elif isinstance(gate, (SqrtXGate, XGate)):
                q = loc[0]
                step = _HALF_PI if isinstance(gate, SqrtXGate) else math.pi
                if axis.get(q) == 'z':
                    flush(q)
                if axis.get(q) == 'x':
                    # Two SX becoming an X, or two X vanishing.
                    angle[q] += step
                else:
                    axis[q] = 'x'
                    angle[q] = step
            elif isinstance(gate, CZGate):
                # Diagonal, so a pending Z rotation passes straight through and
                # is NOT emitted here. That is the whole point: the trailing RZ
                # of one run meets the leading RZ of the next across the CZ.
                for q in loc:
                    if axis.get(q) == 'x':
                        flush(q)
                out.append_gate(gate, op.location)
            else:
                # Anything else is opaque: commit both axes and copy it over.
                for q in loc:
                    flush(q)
                out.append_gate(gate, op.location, op.params)

        for q in list(axis.keys()):
            flush(q)
        return out

    def __repr__(self) -> str:
        """Return a string representation of the pass."""
        return 'NativePhaseFusionPass'

    def __eq__(self, other: Any) -> bool:
        """Two of these passes are interchangeable."""
        return isinstance(other, NativePhaseFusionPass)

    def __hash__(self) -> int:
        """Hash consistent with __eq__."""
        return hash(self.__class__.__name__)
# ===================================================================================
