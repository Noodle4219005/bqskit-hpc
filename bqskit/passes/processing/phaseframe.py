"""This module implements the JointPhaseFrameRetargetPass."""
from __future__ import annotations

import cmath
import logging
import math
import os
from typing import Any

import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cz import CZGate
from bqskit.ir.gates.constant.sx import SqrtXGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.parameterized.rz import RZGate

_logger = logging.getLogger(__name__)

# ==================== HPC: joint phase-frame retargeting ===========================
# The whole of this module. ZXZXZDecomposition decomposes each single-qudit run
# independently and emits RZ-SX-RZ-SX-RZ unconditionally; both halves lose
# gates that exact algebra can see.
#
# Independence. Two locally-minimal decompositions meeting across a CZ still
# carry one redundant rotation, because CZ is diagonal and so commutes with RZ
# on either incident wire:
#
#     [... RZ(c)] CZ_qr [RZ(a) ...]  ==  ... CZ_qr RZ(c + a) ...
#
# Unconditionality. The middle angle is the coordinate on the quotient
# SU(2)/(U(1) x U(1)) and is degenerate in two places:
#
#     run is diagonal      -> zero gates needed, the run is a Z rotation
#     run is anti-diagonal -> one gate needed, RZ(gamma) X
#
# A phase frame alone cannot reach the second point: folding a pending RZ into
# a run leaves the middle angle unchanged, because that angle is invariant
# under Z rotations on either side. Removing an SX therefore requires
# re-deriving the run's unitary, which is why this pass does that rather than
# merging already-emitted gates.
_TWO_PI = 2.0 * math.pi

# Strict on purpose: this pass does exact algebra, so it must never be the
# reason a block drifts toward the success threshold. Anything looser is left
# for the numerical scan to judge.
_TOL = float(os.environ.get('BQSKIT_FRAME_TOL', '1e-12'))


def _wrap(a: float) -> float:
    """Return `a` in (-pi, pi]."""
    return (a + math.pi) % _TWO_PI - math.pi


def _rz(theta: float) -> np.ndarray:
    """Return RZ(theta) as a 2x2 matrix, matching RZGate."""
    return np.diag(
        [np.exp(-0.5j * theta), np.exp(0.5j * theta)],
    ).astype(complex)


def _zxzxz_angles(u: np.ndarray) -> tuple[float, float, float]:
    """Return (lam, theta, phi) with u == RZ(phi) SX RZ(theta) SX RZ(lam).

    The same formula as ZXZXZDecomposition, on a 2x2 rather than a Circuit,
    because this pass needs it per run with the incoming frame already folded
    in and that pass requires a single-qudit Circuit.
    """
    u = np.linalg.det(u) ** (-0.5) * u
    i1 = cmath.phase(u[1, 1])
    i2 = cmath.phase(u[1, 0])
    theta = 2 * np.arctan2(abs(u[1, 0]), abs(u[0, 0])) + math.pi
    phi = i1 + i2 + math.pi
    lam = i1 - i2
    return _wrap(lam), _wrap(theta), _wrap(phi)


class JointPhaseFrameRetargetPass(BasePass):
    """Re-emit every single-qudit run around the fixed diagonal skeleton.

    Carries an unmaterialised Z phase along each wire, so the trailing rotation
    of one run merges with the leading rotation of the next across any number
    of intervening CZ gates, and re-derives each run's unitary so the two
    degenerate cases cost fewer than five gates.

    This is not bit-identical to running without it. The circuit reaching a
    later numerical pass is different, so that pass's seeded random starts
    differ and it may accept a different set of the remaining gates. The
    unitary is preserved exactly; the specific surviving gates are not.
    """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        out = Circuit(circuit.num_qudits, circuit.radixes)
        # pend[q] is the unitary owed on wire q and not yet emitted: a general
        # SU(2) inside a run, collapsing to a pure Z rotation -- the frame --
        # whenever a non-diagonal boundary forces a flush.
        pend: dict[int, np.ndarray] = {}

        def seal(q: int, keep_frame: bool) -> None:
            """Emit wire q's pending unitary.

            keep_frame stops at the trailing Z rotation and leaves it pending.
            That is the mechanism: a CZ cannot see a Z rotation, so carrying it
            past one costs nothing and it merges with the next run.
            """
            u = pend.pop(q, None)
            if u is None:
                return

            if abs(u[1, 0]) <= _TOL and abs(u[0, 1]) <= _TOL:
                # Diagonal: the run is a Z rotation, so no gate is needed.
                theta = _wrap(cmath.phase(u[1, 1]) - cmath.phase(u[0, 0]))
                if abs(theta) <= _TOL:
                    return
                if keep_frame:
                    pend[q] = _rz(theta)
                    return
                out.append_gate(RZGate(), (q,), [theta])
                return

            if abs(u[0, 0]) <= _TOL and abs(u[1, 1]) <= _TOL:
                # Anti-diagonal: u == RZ(gamma) X. X RZ(a) == RZ(-a) X is what
                # lets the leading rotation cross to the far side and join the
                # trailing one, leaving a single X.
                gamma = _wrap(cmath.phase(u[1, 0]) - cmath.phase(u[0, 1]))
                out.append_gate(XGate(), (q,))
                if keep_frame:
                    pend[q] = _rz(gamma)
                    return
                if abs(gamma) > _TOL:
                    out.append_gate(RZGate(), (q,), [gamma])
                return

            lam, theta, phi = _zxzxz_angles(u)
            if abs(lam) > _TOL:
                out.append_gate(RZGate(), (q,), [lam])
            out.append_gate(SqrtXGate(), (q,))
            if abs(theta) > _TOL:
                out.append_gate(RZGate(), (q,), [theta])
            out.append_gate(SqrtXGate(), (q,))
            if keep_frame:
                pend[q] = _rz(phi)
                return
            if abs(phi) > _TOL:
                out.append_gate(RZGate(), (q,), [phi])

        for op in circuit:
            gate = op.gate
            loc = [int(q) for q in op.location]

            if gate.num_qudits == 1 and gate.radixes == (2,):
                q = loc[0]
                u = np.asarray(op.get_unitary().numpy, dtype=complex)
                prev = pend.get(q)
                pend[q] = u if prev is None else u @ prev

            elif isinstance(gate, CZGate):
                # Diagonal, so a pending Z rotation passes straight through and
                # is not emitted here.
                for q in loc:
                    seal(q, keep_frame=True)
                out.append_gate(gate, op.location)

            else:
                # Anything else is opaque: commit everything and copy it over.
                for q in loc:
                    seal(q, keep_frame=False)
                out.append_gate(gate, op.location, op.params)

        for q in list(pend.keys()):
            u = pend[q]
            phase = _wrap(cmath.phase(u[1, 1]) - cmath.phase(u[0, 0]))
            if (
                abs(u[1, 0]) <= _TOL and abs(u[0, 1]) <= _TOL
                and abs(phase) <= _TOL
            ):
                pend.pop(q)
                continue
            seal(q, keep_frame=False)

        circuit.become(out)
        _logger.debug(
            'JointPhaseFrame: %d operations.', circuit.num_operations,
        )

    def __repr__(self) -> str:
        """Return a string representation of the pass."""
        return 'JointPhaseFrameRetargetPass'

    def __eq__(self, other: Any) -> bool:
        """Two of these passes are interchangeable."""
        return isinstance(other, JointPhaseFrameRetargetPass)

    def __hash__(self) -> int:
        """Hash consistent with __eq__."""
        return hash(self.__class__.__name__)
# ===================================================================================
