"""This module implements the LEAPSynthesisPass."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import numpy as np
from scipy.stats import linregress

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.passes.search.frontier import Frontier
from bqskit.passes.search.generator import LayerGenerator
from bqskit.passes.search.generators.seed import SeedLayerGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.passes.search.heuristics import AStarHeuristic
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.runtime import get_runtime
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number

_logger = logging.getLogger(__name__)


# Optional measurement for completed instantiation work that follows LEAP's
# first successful cost evaluation in a map batch. The source-level hooks below
# are deliberately guarded so the ordinary LEAP path remains byte-for-byte
# equivalent in its control flow when the probe is disabled.
_LEAPWASTE_DIR = os.environ.get('BQPROF_LEAPWASTE_DIR') or os.environ.get(
    'BQPROF_LEAPWASTE_AGG_DIR',
)
_LEAPWASTE_AGGREGATE = os.environ.get('BQPROF_LEAPWASTE_AGGREGATE') == '1'
# Optional benchmark-only override; unset preserves the normal LEAP defaults.
_MIN_PREFIX_SIZE_OVERRIDE = os.environ.get('BQSKIT_MIN_PREFIX_SIZE')
_LEAPWASTE_COUNTER = 0
_LEAPWASTE_FH_STATE = {'pid': None, 'fh': None}
_LEAPWASTE_AGG_STATE: dict[str, Any] = {}


def _leapwaste_new_synth_id() -> str:
    """Return a process-qualified identifier for one ``synthesize`` call."""
    global _LEAPWASTE_COUNTER
    _LEAPWASTE_COUNTER += 1
    return f'{os.getpid()}-{_LEAPWASTE_COUNTER}'


def _leapwaste_emit(record: dict[str, Any]) -> None:
    """Append one LEAP iteration record using a fork-safe JSONL handle."""
    if not _LEAPWASTE_DIR or _LEAPWASTE_AGGREGATE:
        return
    try:
        pid = os.getpid()
        if _LEAPWASTE_FH_STATE['pid'] != pid:
            os.makedirs(_LEAPWASTE_DIR, exist_ok=True)
            _LEAPWASTE_FH_STATE['fh'] = open(
                os.path.join(_LEAPWASTE_DIR, f'leapiter_{pid}.jsonl'),
                'a',
                buffering=1,
            )
            _LEAPWASTE_FH_STATE['pid'] = pid
        _LEAPWASTE_FH_STATE['fh'].write(json.dumps(record) + '\n')
    except Exception:
        pass


def _leapwaste_aggregate_flush(force: bool = False) -> None:
    """Atomically refresh one compact aggregate file for this process."""
    if not _LEAPWASTE_DIR or not _LEAPWASTE_AGGREGATE:
        return
    now = time.monotonic()
    if not force and now - _LEAPWASTE_AGG_STATE.get('last_flush', 0.0) < 30.0:
        return
    pid = os.getpid()
    if _LEAPWASTE_AGG_STATE.get('pid') != pid:
        _LEAPWASTE_AGG_STATE.clear()
        _LEAPWASTE_AGG_STATE.update({
            'pid': pid,
            'last_flush': 0.0,
            'iterations': 0,
            'synth_calls': 0,
            'prefix_formed': 0,
            'terminated': 0,
            'sum_n_after_win': 0,
            'n_after_win_count': 0,
            'sum_n_cleared': 0,
            'n_cleared_count': 0,
            'frontier_len_hist': {},
            'n_successors_hist': {},
            'layer_hist': {},
            'max_layer_per_synth_hist': {},
            'active_synth_max_layer': None,
        })
    summary = {
        key: value for key, value in _LEAPWASTE_AGG_STATE.items()
        if key != 'last_flush'
    }
    path = os.path.join(_LEAPWASTE_DIR, f'leapagg_{pid}.json')
    temporary = f'{path}.tmp'
    try:
        os.makedirs(_LEAPWASTE_DIR, exist_ok=True)
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, separators=(',', ':'))
        os.replace(temporary, path)
        _LEAPWASTE_AGG_STATE['last_flush'] = now
    except Exception:
        pass


def _leapwaste_aggregate_record(
    frontier_len: int,
    n_successors: int,
    layer: int,
    prefix_formed: bool,
    terminated: bool,
    n_after_win: int | None,
    n_cleared: int | None,
) -> None:
    """Count one iteration without writing a per-iteration record."""
    if not _LEAPWASTE_DIR or not _LEAPWASTE_AGGREGATE:
        return
    _leapwaste_aggregate_flush()
    hist_frontier = _LEAPWASTE_AGG_STATE['frontier_len_hist']
    hist_successors = _LEAPWASTE_AGG_STATE['n_successors_hist']
    hist_layers = _LEAPWASTE_AGG_STATE['layer_hist']
    frontier_key = str(frontier_len)
    successor_key = str(n_successors)
    layer_key = str(layer)
    hist_frontier[frontier_key] = hist_frontier.get(frontier_key, 0) + 1
    hist_successors[successor_key] = hist_successors.get(successor_key, 0) + 1
    hist_layers[layer_key] = hist_layers.get(layer_key, 0) + 1
    _LEAPWASTE_AGG_STATE['active_synth_max_layer'] = max(
        _LEAPWASTE_AGG_STATE['active_synth_max_layer'] or layer,
        layer,
    )
    _LEAPWASTE_AGG_STATE['iterations'] += 1
    if prefix_formed:
        _LEAPWASTE_AGG_STATE['prefix_formed'] += 1
    if terminated:
        _LEAPWASTE_AGG_STATE['terminated'] += 1
    if n_after_win is not None:
        _LEAPWASTE_AGG_STATE['sum_n_after_win'] += n_after_win
        _LEAPWASTE_AGG_STATE['n_after_win_count'] += 1
    if n_cleared is not None:
        _LEAPWASTE_AGG_STATE['sum_n_cleared'] += n_cleared
        _LEAPWASTE_AGG_STATE['n_cleared_count'] += 1
    _leapwaste_aggregate_flush()


def _leapwaste_aggregate_start() -> None:
    """Start one aggregate process state and count its synthesize call."""
    if not _LEAPWASTE_DIR or not _LEAPWASTE_AGGREGATE:
        return
    _leapwaste_aggregate_flush(force=True)
    _LEAPWASTE_AGG_STATE['synth_calls'] += 1
    _LEAPWASTE_AGG_STATE['active_synth_max_layer'] = 0


def _leapwaste_aggregate_finish() -> None:
    """Persist the latest aggregate state at a synthesize boundary."""
    if _LEAPWASTE_AGG_STATE.get('active_synth_max_layer') is not None:
        max_layer = _LEAPWASTE_AGG_STATE['active_synth_max_layer']
        hist = _LEAPWASTE_AGG_STATE['max_layer_per_synth_hist']
        max_layer_key = str(max_layer)
        hist[max_layer_key] = hist.get(max_layer_key, 0) + 1
        _LEAPWASTE_AGG_STATE['active_synth_max_layer'] = None
    _leapwaste_aggregate_flush(force=True)


class LEAPSynthesisPass(SynthesisPass):
    """
    A pass implementing the LEAP search synthesis algorithm.

    References:
        Ethan Smith, Marc G. Davis, Jeffrey M. Larson, Ed Younis,
        Lindsay Bassman, Wim Lavrijsen, and Costin Iancu. 2022. LEAP:
        Scaling Numerical Optimization Based Synthesis Using an
        Incremental Approach. ACM Transactions on Quantum Computing
        (June 2022). https://doi.org/10.1145/3548693
    """

    def __init__(
        self,
        heuristic_function: HeuristicFunction = AStarHeuristic(),
        layer_generator: LayerGenerator | None = None,
        success_threshold: float = 1e-8,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        max_layer: int | None = None,
        no_progress_layers_allowed: int = 10,
        store_partial_solutions: bool = False,
        partials_per_depth: int = 25,
        min_prefix_size: int = 3,
        instantiate_options: dict[str, Any] = {},
    ) -> None:
        """
        Construct a search-based synthesis pass.

        Args:
            heuristic_function (HeuristicFunction): The heuristic to guide
                search.

            layer_generator (LayerGenerator | None): The successor function
                to guide node expansion. If left as none, then a default
                will be selected before synthesis based on the target
                model's gate set. (Default: None)

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the cost function. (Default: 1e-8)

            cost (CostFunction | None): The cost function that determines
                distance during synthesis. The goal of this synthesis pass
                is to implement circuits for the given unitaries that have
                a cost less than the `success_threshold`.
                (Default: HSDistance())

            max_layer (int): The maximum number of layers to append without
                success before termination. If left as None it will default
                to unlimited. (Default: None)

            no_progress_layers_allowed (int): The maximum number of layers
                allowed without improvement before a warning is issued to
                the user. (Default: 10)

            store_partial_solutions (bool): Whether to store partial solutions
                at different depths inside of the data dict. (Default: False)

            partials_per_depth (int): The maximum number of partials
                to store per search depth. No effect if
                `store_partial_solutions` is False. (Default: 25)

            min_prefix_size (int): The minimum number of layers needed
                to prefix the circuit.

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

        Raises:
            ValueError: If `max_depth` or `min_prefix_size` is nonpositive.
        """
        if _MIN_PREFIX_SIZE_OVERRIDE is not None:
            min_prefix_size = int(_MIN_PREFIX_SIZE_OVERRIDE)

        if not isinstance(heuristic_function, HeuristicFunction):
            raise TypeError(
                'Expected HeursiticFunction, got %s.'
                % type(heuristic_function),
            )

        if layer_generator is not None:
            if not isinstance(layer_generator, LayerGenerator):
                raise TypeError(
                    f'Expected LayerGenerator, got {type(layer_generator)}.',
                )

        if not is_real_number(success_threshold):
            raise TypeError(
                'Expected real number for success_threshold'
                ', got %s' % type(success_threshold),
            )

        if not isinstance(cost, CostFunctionGenerator):
            raise TypeError(
                'Expected cost to be a CostFunctionGenerator, got %s'
                % type(cost),
            )

        if max_layer is not None and not is_integer(max_layer):
            raise TypeError(
                'Expected max_layer to be an integer, got %s' % type(max_layer),
            )

        if max_layer is not None and max_layer <= 0:
            raise ValueError(
                'Expected max_layer to be positive, got %d.' % int(max_layer),
            )

        if min_prefix_size is not None and not is_integer(min_prefix_size):
            raise TypeError(
                'Expected min_prefix_size to be an integer, got %s'
                % type(min_prefix_size),
            )

        if min_prefix_size is not None and min_prefix_size <= 0:
            raise ValueError(
                'Expected min_prefix_size to be positive, got %d.'
                % int(min_prefix_size),
            )

        if not isinstance(instantiate_options, dict):
            raise TypeError(
                'Expected dictionary for instantiate_options, got %s.'
                % type(instantiate_options),
            )

        self.heuristic_function = heuristic_function
        self.layer_gen = layer_generator
        self.success_threshold = success_threshold
        self.cost = cost
        self.max_layer = max_layer
        self.no_progress_layers_allowed = no_progress_layers_allowed
        self.min_prefix_size = min_prefix_size
        self.instantiate_options: dict[str, Any] = {
            'cost_fn_gen': self.cost,
        }
        self.instantiate_options.update(instantiate_options)
        self.store_partial_solutions = store_partial_solutions
        self.partials_per_depth = partials_per_depth

    async def synthesize(
        self,
        utry: UnitaryMatrix | StateVector | StateSystem,
        data: PassData,
    ) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        leapwaste_enabled = bool(_LEAPWASTE_DIR)
        aggregate_enabled = leapwaste_enabled and _LEAPWASTE_AGGREGATE
        synth_id = _leapwaste_new_synth_id() if leapwaste_enabled else None
        iteration = 0
        if aggregate_enabled:
            _leapwaste_aggregate_start()

        # Initialize run-dependent options
        instantiate_options = self.instantiate_options.copy()

        # Seed the PRNG
        if 'seed' not in instantiate_options:
            instantiate_options['seed'] = data.seed

        # Get layer generator for search
        layer_gen = self._get_layer_gen(data)

        # Begin the search with an initial layer
        frontier = Frontier(utry, self.heuristic_function)
        initial_layer = layer_gen.gen_initial_layer(utry, data)
        initial_layer.instantiate(utry, **instantiate_options)
        frontier.add(initial_layer, 0)

        # Track best circuit, initially the initial layer
        best_dist = self.cost.calc_cost(initial_layer, utry)
        best_circ = initial_layer
        best_layer = 0
        best_dists = [best_dist]
        best_layers = [0]
        last_prefix_layer = 0

        # Track partial solutions
        psols: dict[int, list[tuple[Circuit, float]]] = {}

        _logger.debug(f'Search started, initial layer has cost: {best_dist}.')

        # Evalute initial layer
        if best_dist < self.success_threshold:
            _logger.debug('Successful synthesis with 0 layers.')
            if aggregate_enabled:
                _leapwaste_aggregate_finish()
            return initial_layer

        # Record layers that have been warned about
        # to avoid duplicate warnings
        warned_layers: list[int] = []

        # Main loop
        while not frontier.empty():
            if leapwaste_enabled:
                frontier_len_before_pop = len(frontier)
                n_added_this_iter = 0
                prefix_formed = False
                n_cleared = None
            top_circuit, layer = frontier.pop()
            current_iteration = iteration
            iteration += 1

            # Generate successors
            successors = layer_gen.gen_successors(top_circuit, data)

            if len(successors) == 0:
                if leapwaste_enabled:
                    _leapwaste_emit({
                        'synth_id': synth_id,
                        'iteration': current_iteration,
                        'layer': layer,
                        'n_successors': 0,
                        'map_id': None,
                        't_map_start': None,
                        't_map_end': None,
                        'terminated': False,
                        'win_index': None,
                        'n_after_win': None,
                        'frontier_len_before_pop': frontier_len_before_pop,
                        'frontier_len_after_adds': len(frontier),
                        'n_added_this_iter': n_added_this_iter,
                        'prefix_formed': prefix_formed,
                        'n_cleared': n_cleared,
                    })
                if aggregate_enabled:
                    _leapwaste_aggregate_record(
                        frontier_len_before_pop,
                        0,
                        layer,
                        prefix_formed,
                        False,
                        None,
                        n_cleared,
                    )
                continue

            # Instantiate successors
            if leapwaste_enabled:
                t_map_start = time.time()
                map_future = get_runtime().map(
                    Circuit.instantiate,
                    successors,
                    target=utry,
                    **instantiate_options,
                )
                map_id = getattr(map_future, '_bqprof_leapwaste_map_id', None)
                circuits = await map_future
                t_map_end = time.time()
            else:
                circuits = await get_runtime().map(
                    Circuit.instantiate,
                    successors,
                    target=utry,
                    **instantiate_options,
                )

            # Evaluate successors
            for win_index, circuit in enumerate(circuits):
                dist = self.cost.calc_cost(circuit, utry)

                if dist < self.success_threshold:
                    _logger.debug(
                        f'Successful synthesis with {layer + 1} layers.',
                    )
                    if self.store_partial_solutions:
                        data['psols'] = psols
                    if leapwaste_enabled:
                        _leapwaste_emit({
                            'synth_id': synth_id,
                            'iteration': current_iteration,
                            'layer': layer,
                            'n_successors': len(successors),
                            'map_id': map_id,
                            't_map_start': t_map_start,
                            't_map_end': t_map_end,
                            'terminated': True,
                            'win_index': win_index,
                            'n_after_win': len(circuits) - win_index - 1,
                            'frontier_len_before_pop': frontier_len_before_pop,
                            'frontier_len_after_adds': len(frontier),
                            'n_added_this_iter': n_added_this_iter,
                            'prefix_formed': prefix_formed,
                            'n_cleared': n_cleared,
                        })
                    if aggregate_enabled:
                        _leapwaste_aggregate_record(
                            frontier_len_before_pop,
                            len(successors),
                            layer,
                            prefix_formed,
                            True,
                            len(circuits) - win_index - 1,
                            n_cleared,
                        )
                        _leapwaste_aggregate_finish()
                    return circuit

                if self.check_new_best(layer + 1, dist, best_layer, best_dist):
                    plural = '' if layer == 0 else 's'
                    _logger.debug(
                        f'New best circuit found with {layer + 1} layer{plural}'
                        f' and cost: {dist:.12e}.',
                    )
                    best_dist = dist
                    best_circ = circuit
                    best_layer = layer + 1

                    if self.check_leap_condition(
                        layer + 1,
                        best_dist,
                        best_layers,
                        best_dists,
                        last_prefix_layer,
                    ):
                        _logger.debug(f'Prefix formed at {layer + 1} layers.')
                        last_prefix_layer = layer + 1
                        if leapwaste_enabled:
                            prefix_formed = True
                            n_cleared = len(frontier)
                        frontier.clear()
                        if self.max_layer is None or layer + 1 < self.max_layer:
                            frontier.add(circuit, layer + 1)
                            if leapwaste_enabled:
                                n_added_this_iter += 1

                if self.store_partial_solutions:
                    if layer not in psols:
                        psols[layer] = []

                    psols[layer].append((circuit.copy(), dist))

                    if len(psols[layer]) > self.partials_per_depth:
                        psols[layer].sort(key=lambda x: x[1])
                        del psols[layer][-1]

                if self.max_layer is None or layer + 1 < self.max_layer:
                    frontier.add(circuit, layer + 1)
                    if leapwaste_enabled:
                        n_added_this_iter += 1

            if leapwaste_enabled:
                _leapwaste_emit({
                    'synth_id': synth_id,
                    'iteration': current_iteration,
                    'layer': layer,
                    'n_successors': len(successors),
                    'map_id': map_id,
                    't_map_start': t_map_start,
                    't_map_end': t_map_end,
                    'terminated': False,
                    'win_index': None,
                    'n_after_win': None,
                    'frontier_len_before_pop': frontier_len_before_pop,
                    'frontier_len_after_adds': len(frontier),
                    'n_added_this_iter': n_added_this_iter,
                    'prefix_formed': prefix_formed,
                    'n_cleared': n_cleared,
                })
            if aggregate_enabled:
                _leapwaste_aggregate_record(
                    frontier_len_before_pop,
                    len(successors),
                    layer,
                    prefix_formed,
                    False,
                    None,
                    n_cleared,
                )

            layer_diff = abs(best_layer - layer)
            if (
                layer_diff % self.no_progress_layers_allowed == 0
                and layer_diff > 0
                and layer not in warned_layers
            ):
                _logger.warning(
                    'No improvement after '
                    f'{self.no_progress_layers_allowed} layers.',
                )
                warned_layers.append(layer)

        _logger.warning('Frontier emptied.')
        _logger.warning(
            'Returning best known circuit with %d layer%s and cost: %e.'
            % (best_layer, '' if best_layer == 1 else 's', best_dist),
        )
        if self.store_partial_solutions:
            data['psols'] = psols

        if aggregate_enabled:
            _leapwaste_aggregate_finish()
        return best_circ

    def check_new_best(
        self,
        layer: int,
        dist: float,
        best_layer: int,
        best_dist: float,
    ) -> bool:
        """
        Check if the new layer depth and dist are a new best node.

        Args:
            layer (int): The current layer in search.

            dist (float): The current distance in search.

            best_layer (int): The current best layer in the search tree.

            best_dist (float): The current best distance in search.
        """
        better_layer = (
            dist < best_dist
            and (
                best_dist >= self.success_threshold
                or layer <= best_layer
            )
        )
        better_dist_and_layer = (
            dist < self.success_threshold and layer < best_layer
        )
        return better_layer or better_dist_and_layer

    def check_leap_condition(
        self,
        new_layer: int,
        best_dist: float,
        best_layers: list[int],
        best_dists: list[float],
        last_prefix_layer: int,
    ) -> bool:
        """
        Return true if the leap condition is satisfied.

        Args:
            new_layer (int): The current layer in search.

            best_dist (float): The current best distance in search.

            best_layers (list[int]): The list of layers associated
                with recorded best distances.

            best_dists (list[float]): The list of recorded best
                distances.

            last_prefix_layer (int): The last layer a prefix was formed.
        """

        with np.errstate(invalid='ignore', divide='ignore'):
            # Calculate predicted best value
            m, y_int, _, _, _ = linregress(best_layers, best_dists)

        predicted_best = m * (new_layer) + y_int

        # Track new values
        best_layers.append(new_layer)
        best_dists.append(best_dist)

        if np.isnan(predicted_best):
            return False

        # Compute difference between actual value
        delta = predicted_best - best_dist

        _logger.debug(
            'Predicted best value %f for new best best with delta %f.'
            % (predicted_best, delta),
        )

        layers_added = new_layer - last_prefix_layer
        return delta < 0 and layers_added >= self.min_prefix_size

    def _get_layer_gen(self, data: PassData) -> LayerGenerator:
        """
        Set the layer generator.

        If a layer generator has been passed into the constructor, then that
        layer generator will be used. Otherwise, a default layer generator will
        be selected by the gateset.

        If seeds are passed into the data dict, then a SeedLayerGenerator will
        wrap the previously selected layer generator.
        """
        # TODO: Deduplicate this code with qsearch synthesis
        layer_gen = self.layer_gen or data.gate_set.build_mq_layer_generator()

        # Priority given to seeded synthesis
        if 'seed_circuits' in data:
            return SeedLayerGenerator(data['seed_circuits'], layer_gen)

        return layer_gen
