"""This module implements the JointPhaseFrameRetargetPass."""
from __future__ import annotations

import cmath
import json
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


def _rz(theta: float) -> np.ndarray:
    """RZ(theta) as a 2x2, matching RZGate."""
    return np.diag(
        [np.exp(-0.5j * theta), np.exp(0.5j * theta)],
    ).astype(complex)


def _zxzxz_angles(u: np.ndarray) -> tuple[float, float, float]:
    """(lam, theta, phi) with u == RZ(phi) SX RZ(theta) SX RZ(lam)."""
    u = np.linalg.det(u) ** (-0.5) * u
    i1 = cmath.phase(u[1, 1])
    i2 = cmath.phase(u[1, 0])
    theta = 2 * np.arctan2(abs(u[1, 0]), abs(u[0, 0])) + math.pi
    phi = i1 + i2 + math.pi
    lam = i1 - i2
    return _wrap(lam), _wrap(theta), _wrap(phi)


class JointPhaseFrameRetargetPass(BasePass):
    """Re-emit every single-qudit run around the fixed diagonal skeleton.

    ZXZXZDecomposition decomposes each run independently and emits
    RZ-SX-RZ-SX-RZ unconditionally. Both halves lose gates exact algebra sees.

    INDEPENDENCE. Two locally-minimal decompositions meeting across a CZ still
    carry one redundant rotation, because CZ is diagonal and so commutes with
    RZ on either incident wire:

        [... RZ(c)] CZ_qr [RZ(a) ...]  ==  ... CZ_qr RZ(c + a) ...

    So a run's trailing rotation is never emitted; it is carried as an
    unmaterialised frame until a non-diagonal gate seals it.

    UNCONDITIONALITY. The middle angle is the coordinate on the quotient
    SU(2)/(U(1) x U(1)) and is degenerate in two places:

        run is diagonal      -> zero gates, the run IS a Z rotation
        run is anti-diagonal -> one gate, RZ(gamma) X

    A frame alone cannot reach the second: folding a pending RZ into a run
    leaves the middle angle unchanged, since it is invariant under Z rotations
    on either side. Removing an SX therefore needs the run's unitary
    re-derived, which is why this pass does that rather than merging
    already-emitted gates.

    NEVER GROWS. A re-decomposition longer than what it replaces is rejected
    and the original operations are put back. Without that guard a lone SX --
    one gate -- comes back as RZ-SX-RZ-SX plus a frame, the pass grows the
    circuit, and the enclosing ChangePredicate loop never converges.

    NOT BIT-IDENTICAL to running without it: the circuit reaching a later
    numerical pass differs, so that pass's seeded random starts differ and it
    may accept a different set of the remaining gates. The unitary is preserved
    exactly; the specific surviving gates are not.
    """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        stats: dict[str, int] = {
            'ops_in': circuit.num_operations,
            'runs': 0,
            'runs_diagonal': 0,   # zero gates emitted
            'runs_antidiag': 0,   # one gate emitted
            'runs_generic': 0,    # the ordinary case
            'runs_kept': 0,       # re-decomposition rejected as longer
            'sx_saved': 0,
            'frames_crossed_cz': 0,
        }

        out = Circuit(circuit.num_qudits, circuit.radixes)
        # These three are kept SEPARATE on purpose. An earlier version folded
        # the frame into the run's unitary and kept only a flag saying a frame
        # had existed; when a re-decomposition was then rejected and the
        # original operations put back, the frame's angle was unrecoverable and
        # silently dropped, which changed the unitary.
        frame: dict[int, float] = {}                    # owed Z angle
        run_u: dict[int, np.ndarray] = {}               # run's own unitary
        run_ops: dict[int, list[tuple[Any, ...]]] = {}  # its original ops

        def n_sx(ops: list[tuple[Any, ...]]) -> int:
            return sum(1 for g, _, _ in ops if isinstance(g, SqrtXGate))

        def seal(q: int, keep_frame: bool) -> None:
            """Emit wire q's pending frame and run."""
            f = frame.pop(q, 0.0)
            ru = run_u.pop(q, None)
            ops = run_ops.pop(q, [])

            if ru is None:
                if abs(f) <= _TOL:
                    return
                if keep_frame:
                    frame[q] = f
                    return
                out.append_gate(RZGate(), (q,), [f])
                return

            stats['runs'] += 1
            u = ru @ _rz(f)          # the frame applies first
            n_old = len(ops) + (1 if abs(f) > _TOL else 0)

            def put_back() -> None:
                stats['runs_kept'] += 1
                if abs(f) > _TOL:
                    out.append_gate(RZGate(), (q,), [f])
                for g, loc_, params_ in ops:
                    if params_:
                        out.append_gate(g, loc_, params_)
                    else:
                        out.append_gate(g, loc_)

            if abs(u[1, 0]) <= _TOL and abs(u[0, 1]) <= _TOL:
                theta = _wrap(cmath.phase(u[1, 1]) - cmath.phase(u[0, 0]))
                stats['runs_diagonal'] += 1
                stats['sx_saved'] += n_sx(ops)
                if keep_frame:
                    if abs(theta) > _TOL:
                        frame[q] = theta
                    return
                if abs(theta) > _TOL:
                    out.append_gate(RZGate(), (q,), [theta])
                return

            if abs(u[0, 0]) <= _TOL and abs(u[1, 1]) <= _TOL:
                # u == RZ(gamma) X. X RZ(a) == RZ(-a) X lets the leading
                # rotation cross to the far side and join the trailing one.
                gamma = _wrap(cmath.phase(u[1, 0]) - cmath.phase(u[0, 1]))
                n_new = 1 + (0 if keep_frame else int(abs(gamma) > _TOL))
                if n_new > n_old:
                    put_back()
                    return
                stats['runs_antidiag'] += 1
                stats['sx_saved'] += n_sx(ops)
                out.append_gate(XGate(), (q,))
                if keep_frame:
                    if abs(gamma) > _TOL:
                        frame[q] = gamma
                    return
                if abs(gamma) > _TOL:
                    out.append_gate(RZGate(), (q,), [gamma])
                return

            lam, theta, phi = _zxzxz_angles(u)
            n_new = 2 + int(abs(lam) > _TOL) + int(abs(theta) > _TOL)
            if not keep_frame:
                n_new += int(abs(phi) > _TOL)
            if n_new > n_old:
                put_back()
                return
            stats['runs_generic'] += 1
            if abs(lam) > _TOL:
                out.append_gate(RZGate(), (q,), [lam])
            out.append_gate(SqrtXGate(), (q,))
            if abs(theta) > _TOL:
                out.append_gate(RZGate(), (q,), [theta])
            out.append_gate(SqrtXGate(), (q,))
            if keep_frame:
                if abs(phi) > _TOL:
                    frame[q] = phi
                return
            if abs(phi) > _TOL:
                out.append_gate(RZGate(), (q,), [phi])

        for op in circuit:
            gate = op.gate
            loc = [int(q) for q in op.location]

            if gate.num_qudits == 1 and gate.radixes == (2,):
                q = loc[0]
                u = np.asarray(op.get_unitary().numpy, dtype=complex)
                prev = run_u.get(q)
                run_u[q] = u if prev is None else u @ prev
                run_ops.setdefault(q, []).append(
                    (gate, op.location, list(op.params) or None),
                )

            elif isinstance(gate, CZGate):
                # Diagonal, so a pending Z rotation passes straight through and
                # is not emitted here.
                for q in loc:
                    seal(q, keep_frame=True)
                    if q in frame:
                        stats['frames_crossed_cz'] += 1
                out.append_gate(gate, op.location)

            else:
                for q in loc:
                    seal(q, keep_frame=False)
                out.append_gate(gate, op.location, op.params)

        for q in set(list(frame.keys()) + list(run_u.keys())):
            seal(q, keep_frame=False)

        circuit.become(out)
        stats['ops_out'] = circuit.num_operations
        data['joint_phase_frame'] = stats
        # The question this pass exists to answer -- do real post-LEAP circuits
        # contain degenerate runs at all? -- lives in these counters, and
        # PassData does not survive to anywhere they can be aggregated.
        _dir = os.environ.get('BQPROF_SCAN_DIR')
        if _dir:
            with open(f'{_dir}/frame_{os.getpid()}.jsonl', 'a') as fh:
                fh.write(json.dumps(stats) + '\n')
        _logger.debug(
            'JointPhaseFrame: %d -> %d operations.',
            stats['ops_in'], stats['ops_out'],
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
