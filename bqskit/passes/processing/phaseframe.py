"""Joint phase-frame retargeting in the native {CZ, RZ, SX, X} gate set."""
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

_TWO_PI = 2.0 * math.pi
# Strict on purpose: this pass does exact algebra, so it must never be the
# reason a block drifts toward the 1e-8 acceptance threshold. Anything between
# this and 1e-8 is left for the numerical scan to judge.
_TOL = float(os.environ.get('BQSKIT_FRAME_TOL', '1e-12'))


def _wrap(a: float) -> float:
    """Into (-pi, pi]."""
    return (a + math.pi) % _TWO_PI - math.pi


class JointPhaseFrameRetargetPass(BasePass):
    """Re-emit every single-qudit run around the fixed diagonal skeleton.

    WHY THIS EXISTS, and why it is not NativePhaseFusionPass under a new name.

    ZXZXZDecomposition decomposes each single-qudit run INDEPENDENTLY and emits
    RZ-SX-RZ-SX-RZ unconditionally. Both halves of that are lossy:

    1. INDEPENDENTLY. Two locally-minimal decompositions that meet across a CZ
       still carry one redundant rotation, because CZ is diagonal and therefore
       commutes with RZ on either incident wire:

           [... RZ(c_i)] CZ_qr [RZ(a_{i+1}) ...]
             == ... CZ_qr RZ(c_i + a_{i+1}) ...

       Measured on the block that owns the deletion phase of
       square_heisenberg_N16: 26 runs would be 52 SX + 78 RZ, and the survivors
       after gate deletion are 52 SX + 44 RZ. All 34 removals were RZ and none
       was an SX -- the signature of exactly this boundary redundancy.

    2. UNCONDITIONALLY. The middle angle t is the coordinate on the quotient
       SU(2)/(U(1) x U(1)), and it is degenerate in two places that ZXZXZ does
       not special-case. Verified numerically before this pass was written:

           |W_10| == 0  (W diagonal)      <=>  t == +-pi   -> ZERO gates needed
           |W_00| == 0  (W anti-diagonal) <=>  t == 0      -> ONE gate needed (X)

       ZXZXZ emits five in both cases.

    WHAT A FRAME CAN AND CANNOT DO. Folding a pending RZ(phi) into a run
    changes l only; t is invariant under Z rotations on either side (measured:
    0 changes in 400 trials). So a phase frame can NEVER remove an SX. It
    removes the trailing RZ of every run, and item 2 removes the SX -- these
    are separate mechanisms and the pass does both, which is why it strictly
    dominates a post-hoc fusion of already-emitted gates. NativePhaseFusionPass
    merges gates it is given; it cannot re-decompose a run, so it removed zero
    SX on this circuit.

    WHAT IT DOES NOT DO. It is not bit-identical to running without it. The
    circuit reaching the scan is different, so the scan's seeded random starts
    differ, so it may accept a different set of the REMAINING gates. The
    unitary is preserved exactly; the specific surviving gates are not
    guaranteed to match. Verification is therefore `same unitary, and no worse
    1Q/2Q count`, never a digest comparison.
    """

    def __init__(self, collect_stats: bool = True) -> None:
        """Construct the pass; `collect_stats` records per-rule counters."""
        self.collect_stats = collect_stats

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        stats: dict[str, int] = {
            'ops_in': circuit.num_operations,
            'runs': 0,             # single-qudit runs re-emitted
            'runs_identity': 0,    # run + frame was the identity: 0 gates
            'runs_diagonal': 0,    # run + frame was diagonal: 0 gates, frame only
            'runs_antidiag': 0,    # run + frame was anti-diagonal: 1 gate (X)
            'runs_generic': 0,     # the ordinary four-gate case
            'sx_saved': 0,         # SX gates a degenerate case removed
            'frames_crossed_cz': 0,  # frames carried through a CZ unmaterialised
            'frames_dropped': 0,   # trailing frames that vanished mod 2*pi
        }

        out = Circuit(circuit.num_qudits, circuit.radixes)
        # pend[q] is the 2x2 unitary owed on wire q and not yet emitted. It is
        # a general SU(2) while inside a run and collapses to a pure Z rotation
        # (the "frame") whenever a non-diagonal boundary forces a flush.
        pend: dict[int, np.ndarray] = {}

        def seal(q: int, keep_frame: bool) -> None:
            """Emit wire q's pending unitary.

            keep_frame=True stops at the trailing Z rotation and leaves it
            pending, which is the whole point: a CZ cannot see it, so it costs
            nothing to carry it past one and merge it with the next run.
            """
            u = pend.pop(q, None)
            if u is None:
                return
            stats['runs'] += 1

            if abs(u[1, 0]) <= _TOL and abs(u[0, 1]) <= _TOL:
                # Diagonal: the entire run is a Z rotation. Zero gates.
                theta = _wrap(cmath.phase(u[1, 1]) - cmath.phase(u[0, 0]))
                if abs(theta) <= _TOL:
                    stats['runs_identity'] += 1
                else:
                    stats['runs_diagonal'] += 1
                    stats['sx_saved'] += 2
                    if keep_frame:
                        pend[q] = _rz(theta)
                        return
                    out.append_gate(RZGate(), (q,), [theta])
                return

            if abs(u[0, 0]) <= _TOL and abs(u[1, 1]) <= _TOL:
                # Anti-diagonal: u == RZ(gamma) X, so one gate plus a frame.
                # X RZ(a) == RZ(-a) X is what lets the leading rotation be
                # pushed to the far side and merged with the trailing one.
                gamma = _wrap(cmath.phase(u[1, 0]) - cmath.phase(u[0, 1]))
                stats['runs_antidiag'] += 1
                stats['sx_saved'] += 2
                out.append_gate(XGate(), (q,))
                if keep_frame:
                    pend[q] = _rz(gamma)
                    return
                if abs(gamma) > _TOL:
                    out.append_gate(RZGate(), (q,), [gamma])
                return

            stats['runs_generic'] += 1
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
                # is NOT emitted here. That is the mechanism: the trailing
                # rotation of one run meets the leading rotation of the next
                # ACROSS the CZ and the two become one.
                for q in loc:
                    if q in pend:
                        seal(q, keep_frame=True)
                        if q in pend:
                            stats['frames_crossed_cz'] += 1
                out.append_gate(gate, op.location)

            else:
                # Anything else is opaque: commit everything and copy it over.
                for q in loc:
                    seal(q, keep_frame=False)
                out.append_gate(gate, op.location, op.params)

        for q in list(pend.keys()):
            u = pend[q]
            if (
                abs(u[1, 0]) <= _TOL and abs(u[0, 1]) <= _TOL
                and abs(
                    _wrap(cmath.phase(u[1, 1]) - cmath.phase(u[0, 0])),
                ) <= _TOL
            ):
                pend.pop(q)
                stats['frames_dropped'] += 1
                continue
            seal(q, keep_frame=False)

        circuit.become(out)
        stats['ops_out'] = circuit.num_operations
        if self.collect_stats:
            data['joint_phase_frame'] = stats
        # PassData does not survive to anywhere I can aggregate, and the whole
        # question this pass exists to answer -- do real post-LEAP circuits
        # contain degenerate runs at all? -- lives in runs_diagonal /
        # runs_antidiag / sx_saved. So write them where the scan probe already
        # writes, per worker, line-buffered. Silent probes have cost this
        # project four runs; this one fails loudly if the directory is set and
        # unwritable.
        _dir = os.environ.get('BQPROF_SCAN_DIR')
        if _dir:
            import json
            with open(f'{_dir}/frame_{os.getpid()}.jsonl', 'a') as fh:
                fh.write(json.dumps(stats) + '\n')
        _logger.debug(
            'JointPhaseFrame: %d -> %d operations, %d runs '
            '(%d diagonal, %d anti-diagonal), %d SX saved.',
            stats['ops_in'], stats['ops_out'], stats['runs'],
            stats['runs_diagonal'], stats['runs_antidiag'], stats['sx_saved'],
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


def _rz(theta: float) -> np.ndarray:
    """RZ(theta) as a 2x2, matching RZGate."""
    return np.diag(
        [np.exp(-0.5j * theta), np.exp(0.5j * theta)],
    ).astype(complex)


def _zxzxz_angles(u: np.ndarray) -> tuple[float, float, float]:
    """(lam, theta, phi) with u == RZ(phi) SX RZ(theta) SX RZ(lam).

    Same formula as ZXZXZDecomposition, kept here rather than imported because
    that pass takes a whole Circuit and asserts it is single-qudit; this one
    works on the 2x2 directly, per run, with the incoming frame already folded
    in.
    """
    u = np.linalg.det(u) ** (-0.5) * u
    i1 = cmath.phase(u[1, 1])
    i2 = cmath.phase(u[1, 0])
    theta = 2 * np.arctan2(abs(u[1, 0]), abs(u[0, 0])) + math.pi
    phi = i1 + i2 + math.pi
    lam = i1 - i2
    return _wrap(lam), _wrap(theta), _wrap(phi)
