"""This module implements the ForEachBlockPass class."""
from __future__ import annotations

import functools
import json
import logging
import os
import time

import numpy as np
from typing import Any
from typing import Callable

from bqskit.compiler.basepass import _sub_do_work_with_op
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passdata import PassData
from bqskit.compiler.workflow import Workflow
from bqskit.compiler.workflow import WorkflowLike
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.gates.parameterized.pauli import PauliGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.location import CircuitLocation
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.runtime import get_runtime

_logger = logging.getLogger(__name__)

# C0: subdivide ForEachBlockPass's coordinator cost.
#
# docs/16 attributes 23.1% of wall to this pass and has never split it, and
# docs/04 section 9.7 lists that as a known gap. It matters because the four
# candidate stages have four different fixes: serialisation cost wants fewer
# round trips, the preprocessing loop wants offloading to the workers, the
# postprocess loop wants per-block parallelism, and batch_replace is surgery
# on one shared circuit and may simply be irreducible.
#
# Wall and CPU are both recorded for the drain, because the drain's wall time
# includes waiting for workers while its CPU time is the coordinator's own
# serial share -- confusing the two is what makes this stage look bigger than
# it is.
_FOREACH_PROF_DIR = os.environ.get('BQPROF_FOREACH_DIR')

_FOREACH_FH_STATE: dict[str, Any] = {'pid': None, 'fh': None}


def _foreach_emit(record: dict[str, Any]) -> None:
    """Append one ForEachBlockPass phase breakdown."""
    if not _FOREACH_PROF_DIR:
        return
    try:
        pid = os.getpid()
        if _FOREACH_FH_STATE['pid'] != pid:
            os.makedirs(_FOREACH_PROF_DIR, exist_ok=True)
            _FOREACH_FH_STATE['fh'] = open(
                os.path.join(_FOREACH_PROF_DIR, f'foreach_{pid}.jsonl'),
                'a',
                buffering=1,
            )
            _FOREACH_FH_STATE['pid'] = pid
        _FOREACH_FH_STATE['fh'].write(json.dumps(record) + '\n')
    except Exception:
        pass


class ForEachBlockPass(BasePass):
    """
    A pass that executes other passes on each block in the circuit.

    This is a control pass that executes a workflow on every block in the
    circuit. This will be done in parallel.
    """

    key = 'ForEachBlockPass_data'
    """The key in data, where block data will be put."""

    pass_down_key_prefix = 'ForEachBlockPass_pass_down_'
    """If a key exists in the pass data with this prefix, pass it to blocks."""

    pass_down_block_specific_key_prefix = (
        'ForEachBlockPass_specific_pass_down_'
    )
    """
    Data specific to the processing of individual blocks in a partitioned
    circuit can be injected into the `PassData` in `run` by using this prefix.

    The expected type of the associated value is `dict[int, Any]`, where
    integer (sub-)keys correspond to block numbers in a partitioned quantum
    circuit.

    Pseudocode example for seed circuits:
        seeds = {block_id: [seed_circuit_a, seed_circuit_b, ...], ...}
        key = self.pass_down_block_specific_key_prefix + 'seed_circuits'
        seed_updater = UpdateDataPass(key, seeds)
        workflow = Workflow([..., seed_updater, ForEachBlockPass(...), ...])
    """

    def __init__(
        self,
        loop_body: WorkflowLike,
        calculate_error_bound: bool = False,
        collection_filter: Callable[[Operation], bool] | None = None,
        replace_filter: ReplaceFilterFn | str = 'always',
        batch_size: int | None = None,
    ) -> None:
        """
        Construct a ForEachBlockPass.

        Args:
            loop_body (WorkflowLike): The workflow to execute on every block.

            calculate_error_bound (bool): If set to true, will calculate
                errors on blocks after running `loop_body` on them and
                use these block errors to calculate an upper bound on the
                full circuit error. (Default: False)

            collection_filter (Callable[[Operation], bool] | None):
                A predicate that determines which operations should have
                `loop_body` called on them. Called with each operation
                in the circuit. If this returns true, that operation will
                be formed into an individual circuit and passed through
                `loop_body`. Defaults to all CircuitGates,
                ConstantUnitaryGates, and VariableUnitaryGates.
                #TODO: address importability

            replace_filter (ReplaceFilterFn | str | None):
                A predicate that determines if the resulting circuit, after
                calling `loop_body` on a block, should replace the original
                operation. Called with the circuit output from `loop_body`
                and the original operation. If this returns true, the
                operation will be replaced with the new circuit.
                Defaults to always replace. If none is passed, will
                generate a replace filter always replaces. If a string is
                passed, will generate a replace filter corresponding to
                the string. The string should either be 'always', 'less-than',
                'less-than-multi', 'less-than-many', 'less-than-respecting',
                'less-than-respecting-multi', or 'less-than-respecting-many'.
                    - 'always' will always replace
                    - 'less-than' will replace if the new circuit has fewer
                        gates than the old circuit.
                    - 'less-than-multi' will replace if the new circuit has
                        fewer multi-qudit gates than the old circuit.
                    - 'less-than-many' will replace if the new circuit has
                        fewer many-qudit gates than the old circuit.
                    - 'less-than-respecting' will replace if the new circuit
                        has fewer gates than the old circuit or the old
                        doesn't respect the model (ignoring single-qudit
                        gate sets).
                    - 'less-than-respecting-multi' will replace if the new
                        circuit has fewer multi-qudit gates than the old
                        circuit or the old doesn't respect the model
                        (ignoring single-qudit gate sets).
                    - 'less-than-respecting-many' will replace if the new
                        circuit has fewer many-qudit gates than the old
                        circuit or the old doesn't respect the model
                        (ignoring single-qudit gate sets).
                    - 'less-than-respecting-fully' will replace if the new
                        circuit has fewer gates than the old circuit or
                        the old doesn't respect the model.
                    - 'less-than-respecting-fully-multi' will replace if
                        the new circuit has fewer multi-qudit gates than
                        the old circuit or the old doesn't respect the model.
                    - 'less-than-respecting-fully-many' will replace if
                        the new circuit has fewer many-qudit gates than
                        the old circuit or the old doesn't respect the model.
                Defaults to 'always'.  #TODO: address importability

            batch_size (int): (Deprecated).
        """
        if batch_size is not None:
            import warnings
            warnings.warn(
                'Batch size is no longer supported, this warning will'
                ' become an error in a future update.',
                DeprecationWarning,
            )

        threshold_text = os.environ.get('BQSKIT_BLOCK_ERROR_THRESHOLD')
        if threshold_text is None:
            self.error_threshold = None
        else:
            try:
                self.error_threshold = float(threshold_text)
            except ValueError as err:
                raise ValueError(
                    'BQSKIT_BLOCK_ERROR_THRESHOLD must be a float, got '
                    f'{threshold_text!r}.',
                ) from err

        self.calculate_error_bound = calculate_error_bound
        if self.error_threshold is not None:
            self.calculate_error_bound = True
        self.collection_filter = collection_filter or default_collection_filter
        self.replace_filter = replace_filter or default_replace_filter
        self.workflow = Workflow(loop_body)

        if not callable(self.collection_filter):
            raise TypeError(
                'Expected callable method that maps Operations to booleans for'
                f' collection_filter, got {type(self.collection_filter)}.',
            )

        if not isinstance(self.replace_filter, str):
            if not callable(self.replace_filter):
                raise TypeError(
                    'Expected either string representing a valid replacement'
                    ' filter or callable method that maps Circuit and'
                    ' Operations to bools for replace_filter'
                    f' , got {type(self.replace_filter)}.',
                )

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        # Get the callable replacement filter
        if isinstance(self.replace_filter, str):
            method = self.replace_filter
            replace_filter = gen_replace_filter(method, data.model)
        else:
            replace_filter = self.replace_filter

        # Make room in data for block data
        if self.key not in data:
            data[self.key] = []

        _t_pass_start = time.perf_counter()
        _c_pass_start = time.process_time()

        # Collect blocks
        blocks: list[tuple[int, Operation]] = []
        for cycle, op in circuit.operations_with_cycles():
            if self.collection_filter(op):
                blocks.append((cycle, op))

        # No blocks, no work
        if len(blocks) == 0:
            data[self.key].append([])
            return

        _t_collect_end = time.perf_counter()

        # Get the machine model
        model = data.model
        coupling_graph = data.connectivity

        # Preprocess blocks
        submodels: list[MachineModel] = []
        subnumberings: list[dict[int, int]] = []
        pass_down_datas: list[dict[str, Any]] = []
        cycles: list[int] = []
        _t_preprocess_copy = 0.0
        _t_preprocess_submodel = 0.0
        _t_preprocess_passdata = 0.0
        for i, (cycle, op) in enumerate(blocks):

            # Form Submodel
            if _FOREACH_PROF_DIR:
                _t0 = time.perf_counter()
            subradixes = [circuit.radixes[q] for q in op.location]
            subnumbering = {op.location[i]: i for i in range(len(op.location))}
            submodel = MachineModel(
                len(op.location),
                coupling_graph.get_subgraph(op.location, subnumbering),
                model.gate_set,
                subradixes,
            )
            if _FOREACH_PROF_DIR:
                _t_preprocess_submodel += time.perf_counter() - _t0

            # Pass down data that is indexed by block remains a per-task
            # ingredient; the worker creates the PassData object.
            pass_down_data: dict[str, Any] = {}
            for key in data:
                if key.startswith(self.pass_down_key_prefix):
                    pass_down_data[key] = data[key]
                elif (
                    key.startswith(self.pass_down_block_specific_key_prefix)
                    and i in data[key]
                ):
                    pass_down_data[key] = data[key][i]

            submodels.append(submodel)
            subnumberings.append(subnumbering)
            pass_down_datas.append(pass_down_data)
            cycles.append(cycle)

        # Do the work
        #
        # Results are consumed incrementally rather than with a single
        # `await map(...)` barrier. Per-block postprocessing -- the
        # replace_filter call and building the replacement Operation -- is
        # pure Python running in this one worker, while the other workers are
        # still synthesising later blocks. Draining with `next` lets that
        # bookkeeping overlap the synthesis instead of queueing behind it.
        #
        # With the filter disabled, this is a scheduling change only: every
        # block is still postprocessed exactly once, results are stored by
        # their original index, and `batch_replace` still receives points and
        # ops in block order below. A block below the configured threshold is
        # intentionally left with no replacement.
        _t_preprocess_end = time.perf_counter()

        n2q_per_block: list[float | int] = []
        for _, op in blocks:
            block_circuit = getattr(op.gate, '_circuit', None)
            if block_circuit is None or len(op.location) < 2:
                # Infinity means ALWAYS DISPATCH, and the width guard is not a
                # detail. `ForEachBlockPass` wraps very different passes in this
                # workflow -- one of them is `ZXZXZDecomposition`, a REBASE over
                # single-qubit blocks. A 1-qubit block holds 0 two-qubit gates
                # by definition, so any threshold >= 1 would skip every one of
                # them and emit a circuit still carrying gates outside the
                # target gate set. That is not a quality trade, it is an invalid
                # circuit.
                #
                # The question this filter asks -- "is there enough 2Q structure
                # for resynthesis to pay?" -- is meaningless below 2 qudits, so
                # declining to answer it is correct behaviour, not a special
                # case.
                n2q_per_block.append(float('inf'))
            else:
                n2q_per_block.append(
                    sum(1 for o in block_circuit if o.num_qudits >= 2),
                )
        dispatch_idx = list(range(len(blocks)))
        n_dispatch = len(dispatch_idx)
        if n_dispatch > 0:
            future = get_runtime().map(
                _sub_do_work_with_op,
                [self.workflow] * n_dispatch,
                [blocks[i][1] for i in dispatch_idx],
                [submodels[i] for i in dispatch_idx],
                [subnumberings[i] for i in dispatch_idx],
                [cycles[i] for i in dispatch_idx],
                [self.calculate_error_bound] * n_dispatch,
                [data.seed] * n_dispatch,
                [pass_down_datas[i] for i in dispatch_idx],
                cost_hints=[
                    float(4 ** len(blocks[i][1].location))
                    for i in dispatch_idx
                ],
            )

        _t_dispatch_end = time.perf_counter()

        # Emit the coordinator's own cost as soon as it is paid, not at the
        # end of the pass. Everything above this line -- collect, the
        # per-block deep copy / submodel / PassData loop, and the map()
        # serialisation -- is finished and measured here, while the drain
        # below may never finish at all: at max_synthesis_size 4 and 5 the
        # synthesis pass routinely exceeds any budget we can give it, and a
        # record written only at pass end is then never written.
        #
        # That would make the coordinator cost unmeasurable in exactly the
        # regime the question is about, since the whole point of the msz
        # sweep is that per-block marshalling grows with block width.
        # 'phase' separates the two records; reducers must select one.
        _foreach_emit({
            'phase': 'dispatch',
            'n_blocks': len(blocks),
            'n_dispatch': n_dispatch,
            'n_skipped': len(blocks) - n_dispatch,
            'n2q_per_block': n2q_per_block,
            't_collect': round(_t_collect_end - _t_pass_start, 6),
            't_preprocess': round(_t_preprocess_end - _t_collect_end, 6),
            't_dispatch': round(_t_dispatch_end - _t_preprocess_end, 6),
        })

        _postprocess_cpu = 0.0

        num_blocks = len(blocks)
        completed_subcircuits: list[Circuit] = [None] * num_blocks  # type: ignore
        completed_block_datas: list[PassData] = [None] * num_blocks  # type: ignore
        # Postprocessed replacement for each block, or None if the block is
        # not being replaced. Indexed by block so order is independent of
        # the order results happen to arrive in.
        replacements: list[tuple[CircuitPoint, Operation] | None]
        replacements = [None] * num_blocks
        error_sum = 0.0
        n_error_rejected = 0
        max_block_error = 0.0
        num_remaining = n_dispatch

        while num_remaining > 0:
            _fetched = await get_runtime().next(future)
            _t_chunk = time.perf_counter()
            for index, result in _fetched:
                subcircuit, block_data = result
                num_remaining -= 1
                block_index = dispatch_idx[index]
                completed_subcircuits[block_index] = subcircuit
                completed_block_datas[block_index] = block_data

                if self.calculate_error_bound:
                    max_block_error = max(max_block_error, block_data.error)

                cycle, op = blocks[block_index]

                # Mark Blocks to be Replaced
                if replace_filter(subcircuit, op):
                    # `not error <= threshold` rather than `error > threshold`
                    # so that a NaN error is REJECTED. Every comparison
                    # against NaN is False, so the natural spelling would let
                    # a block through precisely when its distance could not be
                    # computed -- the one case where accepting it is least
                    # defensible.
                    if (
                        self.error_threshold is not None
                        and not block_data.error <= self.error_threshold
                    ):
                        n_error_rejected += 1
                        _logger.warning(
                            'Block %d rejected by error threshold: measured '
                            'error %g exceeds threshold %g.',
                            block_index,
                            block_data.error,
                            self.error_threshold,
                        )
                        # Emitted per rejection, not only in the 'complete'
                        # summary. The summary is written when the pass ends,
                        # and the runs where this fires are exactly the ones
                        # that time out inside the pass and never get there --
                        # the first tokyo attempt recorded one 'dispatch' line
                        # and nothing else.
                        _foreach_emit({
                            'phase': 'error_reject',
                            'block': block_index,
                            'error': block_data.error,
                            'threshold': self.error_threshold,
                        })
                        block_data['replaced'] = False
                        continue

                    _logger.debug(f'Replacing block {block_index}.')
                    replacements[block_index] = (
                        CircuitPoint(cycle, op.location[0]),
                        Operation(
                            CircuitGate(subcircuit, True),
                            op.location,
                            subcircuit.params,
                        ),
                    )
                    block_data['replaced'] = True

                    # Calculate Error
                    error_sum += block_data.error
                else:
                    block_data['replaced'] = False
            _postprocess_cpu += time.perf_counter() - _t_chunk

        _t_drain_end = time.perf_counter()

        points: list[CircuitPoint] = []
        ops: list[Operation] = []
        for replacement in replacements:
            if replacement is not None:
                points.append(replacement[0])
                ops.append(replacement[1])

        # Replace blocks
        _t_replace_start = time.perf_counter()
        circuit.batch_replace(points, ops)
        _t_replace_end = time.perf_counter()

        complete_record: dict[str, Any] = {
            'phase': 'complete',
            'n_blocks': num_blocks,
            'n_replaced': len(points),
            'n_error_rejected': n_error_rejected,
            'error_threshold': self.error_threshold,
            'wall': round(_t_replace_end - _t_pass_start, 6),
            'cpu': round(time.process_time() - _c_pass_start, 6),
            # collect: scan the circuit and filter for blocks
            't_collect': round(_t_collect_end - _t_pass_start, 6),
            # preprocess: per-block deep copy, submodel, PassData
            't_preprocess': round(_t_preprocess_end - _t_collect_end, 6),
            't_preprocess_copy': round(_t_preprocess_copy, 6),
            't_preprocess_submodel': round(_t_preprocess_submodel, 6),
            't_preprocess_passdata': round(_t_preprocess_passdata, 6),
            # dispatch: the map() call itself, i.e. serialisation
            't_dispatch': round(_t_dispatch_end - _t_preprocess_end, 6),
            # drain wall INCLUDES waiting on workers ...
            't_drain_wall': round(_t_drain_end - _t_dispatch_end, 6),
            # ... while this is the coordinator's own share of it
            't_postprocess': round(_postprocess_cpu, 6),
            # replace: surgery on the one shared circuit
            't_replace': round(_t_replace_end - _t_replace_start, 6),
        }
        if self.calculate_error_bound:
            complete_record['max_block_error'] = max_block_error
        _foreach_emit(complete_record)

        # Record block data into pass data
        data[self.key].append(completed_block_datas)

        # Record error
        data.update_error_mul(error_sum)
        if self.calculate_error_bound:
            _logger.debug(f'New circuit error is {data.error}.')


def default_collection_filter(op: Operation) -> bool:
    return isinstance(
        op.gate, (
            CircuitGate,
            ConstantUnitaryGate,
            VariableUnitaryGate,
            PauliGate,
        ),
    )


def default_replace_filter(circuit: Circuit, op: Operation) -> bool:
    """Always replace."""
    # legacy name and style for backwards compatibility
    return True


def _less_than(new: Circuit, old: Operation) -> bool:
    """Return true if the new circuit has fewer gates."""
    if isinstance(old.gate, CircuitGate):
        return new.num_operations < old.gate._circuit.num_operations

    return True  # TODO: Re-evaluate always true when old is not a circuit


def _less_than_multi(new: Circuit, old: Operation) -> bool:
    """Return true if the new circuit has fewer multi-qudit gates."""
    if isinstance(old.gate, CircuitGate):
        org = old.gate._circuit
        omq = sum([c for g, c in org.gate_counts.items() if g.num_qudits > 1])
        osq = sum([c for g, c in org.gate_counts.items() if g.num_qudits == 1])
        nmq = sum([c for g, c in new.gate_counts.items() if g.num_qudits > 1])
        nsq = sum([c for g, c in new.gate_counts.items() if g.num_qudits == 1])
        return (nmq, nsq) < (omq, osq)

    return True


def _less_than_many(new: Circuit, old: Operation) -> bool:
    """Return true if the new circuit has fewer many-qudit gates."""
    if isinstance(old.gate, CircuitGate):
        org = old.gate._circuit
        omq = sum([c for g, c in org.gate_counts.items() if g.num_qudits > 2])
        otq = sum([c for g, c in org.gate_counts.items() if g.num_qudits == 2])
        osq = sum([c for g, c in org.gate_counts.items() if g.num_qudits == 1])
        nmq = sum([c for g, c in new.gate_counts.items() if g.num_qudits > 2])
        ntq = sum([c for g, c in new.gate_counts.items() if g.num_qudits == 2])
        nsq = sum([c for g, c in new.gate_counts.items() if g.num_qudits == 1])
        return (nmq, ntq, nsq) < (omq, otq, osq)

    return True


def _is_respecting(
    circuit: Circuit,
    location: CircuitLocation,
    model: MachineModel,
    fully: bool = False,
) -> bool:
    """
    Return true if the `circuit` respects the `model` at `location`.

    Args:
        circuit (Circuit): The circuit to check.

        location (CircuitLocation): The location to check.

        model (MachineModel): The machine model to check against.

        fully (bool): If set to true, will check if the circuit respects
            the model fully. If set to false, will ignore single-qudit
            gate sets. (Default: False)

    Returns:
        True if the circuit respects the model at the location. This implies
        that the circuit can be run on the machine at the location.
    """
    org_mq_gates = circuit.gate_set.multi_qudit_gates
    org_sq_gates = circuit.gate_set.single_qudit_gates

    if any(g not in model.gate_set for g in org_mq_gates):
        return False

    if fully and any(g not in model.gate_set for g in org_sq_gates):
        return False

    if any(
        (location[e[0]], location[e[1]]) not in model.coupling_graph
        for e in circuit.coupling_graph
    ):
        return False

    return True


def _less_than_fn_respecting(
    new: Circuit,
    old: Operation,
    model: MachineModel,
    fn: ReplaceFilterFn,
) -> bool:
    """Return true if the new circuit has fewer gates or the old doesn't respect
    the model."""
    if isinstance(old.gate, CircuitGate):
        if not _is_respecting(old.gate._circuit, old.location, model):
            if not _is_respecting(new, old.location, model):
                _logger.debug("New block doesn't respect model.")
            return True

        if not _is_respecting(new, old.location, model):
            _logger.debug("New block doesn't respect model.")
            return False

    return fn(new, old)


def _less_than_fn_respecting_fully(
    new: Circuit,
    old: Operation,
    model: MachineModel,
    fn: ReplaceFilterFn,
) -> bool:
    """Return true if the new circuit has fewer gates or the old doesn't respect
    the model."""
    if isinstance(old.gate, CircuitGate):
        if not _is_respecting(old.gate._circuit, old.location, model, True):
            if not _is_respecting(new, old.location, model, True):
                _logger.debug("New block doesn't respect model.")
            return True

        if not _is_respecting(new, old.location, model, True):
            _logger.debug("New block doesn't respect model.")
            return False

    return fn(new, old)


def gen_always(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that always replaces."""
    # legacy name and style for backwards compatibility
    return default_replace_filter


def gen_less_than(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    gates."""
    return _less_than


def gen_less_than_multi(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    multi-qudit gates."""
    return _less_than_multi


def gen_less_than_many(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    many-qudit gates."""
    return _less_than_many


def gen_less_than_rspt(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    gates or the old doesn't respect the model."""
    return functools.partial(
        _less_than_fn_respecting,
        model=model,
        fn=_less_than,
    )


def gen_less_than_rspt_multi(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    multi-qudit gates or the old doesn't respect the model."""
    return functools.partial(
        _less_than_fn_respecting,
        model=model,
        fn=_less_than_multi,
    )


def gen_less_than_rspt_many(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    many-qudit gates or the old doesn't respect the model."""
    return functools.partial(
        _less_than_fn_respecting,
        model=model,
        fn=_less_than_many,
    )


def gen_less_than_rspt_fully(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    gates or the old doesn't respect the model."""
    return functools.partial(
        _less_than_fn_respecting_fully,
        model=model,
        fn=_less_than,
    )


def gen_less_than_rspt_fully_multi(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    multi-qudit gates or the old doesn't respect the model."""
    return functools.partial(
        _less_than_fn_respecting_fully,
        model=model,
        fn=_less_than_multi,
    )


def gen_less_than_rspt_fully_many(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    many-qudit gates or the old doesn't respect the model."""
    return functools.partial(
        _less_than_fn_respecting_fully,
        model=model,
        fn=_less_than_many,
    )


def gen_replace_filter(method: str, model: MachineModel) -> ReplaceFilterFn:
    """
    Generate a replace filter for use during the standard workflow.

    Args:
        method (str): The method to use for the replace filter. See
            :class:`ForEachBlockPass` for more information.

        model (MachineModel): The machine model to potentially respect.

    Returns:
        A replace filter function.
    """
    replace_filters = {
        'always': gen_always,
        'less-than': gen_less_than,
        'less-than-multi': gen_less_than_multi,
        'less-than-many': gen_less_than_many,
        'less-than-respecting': gen_less_than_rspt,
        'less-than-respecting-multi': gen_less_than_rspt_multi,
        'less-than-respecting-many': gen_less_than_rspt_many,
        'less-than-respecting-fully': gen_less_than_rspt_fully,
        'less-than-respecting-fully-multi': gen_less_than_rspt_fully_multi,
        'less-than-respecting-fully-many': gen_less_than_rspt_fully_many,
    }

    if method not in replace_filters:
        raise ValueError(f'Unknown replace filter method {method}.')

    return replace_filters[method](model)


ReplaceFilterFn = Callable[[Circuit, Operation], bool]


class ClearAllBlockData(BasePass):
    """Clear all block data and passed down data from the pass data."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        for key in list(data.keys()):
            if key.startswith(ForEachBlockPass.key):
                del data[key]
            elif key.startswith(ForEachBlockPass.pass_down_key_prefix):
                del data[key]
