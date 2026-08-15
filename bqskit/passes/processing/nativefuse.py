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

_TWO_PI = 2.0 * math.pi
_HALF_PI = math.pi / 2.0

# Strict on purpose. This pass does ALGEBRA, not approximate synthesis, so it
# must never be the reason a block drifts toward the 1e-8 success threshold.
# Anything between this and 1e-8 is left for the numerical scan to judge.
_TOL = float(os.environ.get('BQSKIT_FUSE_TOL', '1e-12'))


class NativePhaseFusionPass(BasePass):
    """Fuse Z- and X-axis rotations exactly, in the native gate set.

    WHY THIS EXISTS. ZXZXZDecomposition expands every single-qudit run into
    RZ-SX-RZ-SX-RZ unconditionally -- there is no special case for an angle
    that happens to be zero, and no attempt to look past the two-qudit gate
    that ends the run. Gate deletion then runs AFTER that expansion and pays a
    full numerical re-instantiation per candidate to rediscover that some of
    those RZ are redundant.

    Measured on square_heisenberg_N16, msz=4, the block that owns the deletion
    phase (129.5 s of a 176 s phase): 142 candidates, 34 accepted removals,
    ALL 34 of them RZ and not one SX. Its gate counts, 52 SX / 44 RZ kept and
    12 CZ, are exactly 26 five-gate runs with 34 RZ taken out -- the signature
    of a phase gauge, not of a numerical discovery.

    THE IDENTITIES, all exact equalities and all verified numerically:

        RZ(a) RZ(b) = RZ(a+b)                 exact
        [CZ, RZ (x) I] = 0                    both are diagonal
        RZ(2*pi) = -I                          a global phase
        SX SX = X                              exact, max elementwise diff 0.0
        X X = I                                exact
        CZ does NOT commute with SX            checked, and relied upon

    Dropping the 2*pi is sound because the distance BQSKit judges by is
    sqrt[D]{1 - |Tr(U1' U2)|^D / N^D} (UnitaryMatrix.get_distance_from): the
    absolute value makes it blind to global phase.

    WHAT IT DOES NOT DO. It is not bit-identical to running without it. The
    circuit that reaches the scan is different, so the scan's seeded random
    starts differ, so it may accept a different set of the REMAINING gates.
    The unitary is preserved exactly; the specific surviving gates are not
    guaranteed to match. Verification therefore has to be `same unitary, and
    no worse 1Q/2Q count`, never a digest comparison.

    It also finds only the LOCAL redundancy. On adder_8 the same accounting
    bounds it at 37.1% of accepted removals against 60.2% on
    square_heisenberg; the rest needs the whole block's 4^w - 1 parameters to
    be re-fitted and no rewriting can see it.
    """

    def __init__(self, collect_stats: bool = True) -> None:
        """Construct the pass; `collect_stats` records per-rule counters."""
        self.collect_stats = collect_stats

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        stats: dict[str, int] = {
            'ops_in': circuit.num_operations,
            'rz_fused': 0,          # an RZ absorbed into another RZ
            'rz_zero': 0,           # an RZ whose accumulated angle vanished
            'rz_through_cz': 0,     # fusions that crossed a CZ
            'sx_to_x': 0,           # SX SX collapsed to one X
            'x_cancelled': 0,       # X X vanished
            'sweeps': 0,
        }

        # Alternate until nothing changes. The two axes feed each other: a
        # vanishing RZ leaves SX-SX adjacent, which becomes X, and two X
        # become nothing, which can leave two RZ adjacent again.
        while True:
            stats['sweeps'] += 1
            before = circuit.num_operations
            new_circuit = self._sweep(circuit, stats)
            circuit.become(new_circuit)
            if circuit.num_operations >= before:
                break

        stats['ops_out'] = circuit.num_operations
        if self.collect_stats:
            data['native_phase_fusion'] = stats
        _logger.debug(
            'NativePhaseFusion: %d -> %d operations in %d sweeps.',
            stats['ops_in'], stats['ops_out'], stats['sweeps'],
        )

    def _sweep(self, circuit: Circuit, stats: dict[str, int]) -> Circuit:
        """One left-to-right pass, rebuilding the circuit."""
        out = Circuit(circuit.num_qudits, circuit.radixes)

        # Per qudit: the axis with an unemitted rotation, and its angle.
        # At most one axis can be pending at a time, because meeting a gate on
        # the other axis flushes this one first -- Z and X do not commute.
        axis: dict[int, str] = {}
        angle: dict[int, float] = {}
        # Whether the pending Z rotation has already passed a CZ. Only used to
        # tell a plain neighbour-fusion from one the CZ commutation enabled,
        # which is the number that says whether this pass is doing anything
        # the old adjacent-run merge could not.
        crossed: dict[int, bool] = {}

        def flush(q: int) -> None:
            a = axis.get(q)
            if a is None:
                return
            t = angle[q]
            axis.pop(q)
            angle.pop(q)
            was_crossed = crossed.pop(q, False)
            if a == 'z':
                # Into (-pi, pi]. RZ(2*pi) is -I, i.e. global phase.
                t = (t + math.pi) % _TWO_PI - math.pi
                if abs(t) <= _TOL:
                    stats['rz_zero'] += 1
                    return
                out.append_gate(RZGate(), (q,), [t])
                if was_crossed:
                    stats['rz_through_cz'] += 1
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
                    stats['rz_fused'] += 1
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
                    if abs(step - _HALF_PI) < _TOL:
                        stats['sx_to_x'] += 1
                    else:
                        stats['x_cancelled'] += 1
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
                    elif axis.get(q) == 'z':
                        crossed[q] = True
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
