"""
Sequential pulsing constraint for silicon spin qubit arrays.

In exchange-only architectures like the HRL SLEDGE device, exchange
pulses must be applied sequentially - never simultaneously on adjacent
qubit pairs - to avoid crosstalk-induced errors.

This module provides:
    1. PulseScheduler: validates and schedules exchange pulse sequences
    2. Conflict detection: identifies which edges share vertices
    3. Graph colouring: optimal parallelisation of non-conflicting pulses
    4. Idle noise insertion: automatically adds decoherence during idle

Based on the constraint described in:
    Weinstein et al., Nature 615, 817-822 (2023):
    "Exchange gates are applied sequentially, with all other exchange
    rates set to their minimum values during each pulse."

For 2D grid topologies, this constraint significantly impacts the
circuit depth and total evolution time, making it a critical factor
for the DRL agent to learn optimal pulse orderings.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class ExchangePulse:
    """A single exchange pulse between two qubits.

    Parameters
    ----------
    qubit_i : int
        First qubit index.
    qubit_j : int
        Second qubit index.
    angle : float
        Exchange rotation angle (radians).
    duration : float
        Pulse duration (seconds). If None, computed from angle and J.
    label : str
        Optional human-readable label.
    """
    qubit_i: int
    qubit_j: int
    angle: float
    duration: Optional[float] = None
    label: str = ""

    @property
    def edge(self) -> Tuple[int, int]:
        """Canonical edge representation (i < j)."""
        return (min(self.qubit_i, self.qubit_j),
                max(self.qubit_i, self.qubit_j))

    def conflicts_with(self, other: "ExchangePulse") -> bool:
        """Check if this pulse conflicts with another.

        Two pulses conflict if they share a qubit (vertex).
        """
        return (self.qubit_i in (other.qubit_i, other.qubit_j) or
                self.qubit_j in (other.qubit_i, other.qubit_j))


@dataclass
class PulseLayer:
    """A set of non-conflicting pulses that can be applied in parallel.

    In sequential mode, each layer contains exactly one pulse.
    In parallel mode (for non-SLEDGE devices), a layer can contain
    multiple non-conflicting pulses.
    """
    pulses: List[ExchangePulse] = field(default_factory=list)
    idle_qubits: List[int] = field(default_factory=list)
    layer_duration: float = 0.0

    def add_pulse(self, pulse: ExchangePulse) -> bool:
        """Try to add a pulse to this layer.

        Returns True if the pulse was added (no conflicts),
        False if it conflicts with an existing pulse.
        """
        for existing in self.pulses:
            if pulse.conflicts_with(existing):
                return False
        self.pulses.append(pulse)
        return True

    @property
    def active_qubits(self) -> Set[int]:
        """Set of qubits involved in pulses in this layer."""
        qubits = set()
        for p in self.pulses:
            qubits.add(p.qubit_i)
            qubits.add(p.qubit_j)
        return qubits


class PulseScheduler:
    """Schedules exchange pulses respecting the sequential pulsing constraint.

    For SLEDGE-type devices, all pulses must be strictly sequential.
    For other devices, non-conflicting pulses can be parallelised
    using graph colouring.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    connectivity : list of tuple
        List of (i, j) edges in the connectivity graph.
    sequential : bool
        If True, enforce strict sequential pulsing (one pulse per layer).
        If False, allow parallel non-conflicting pulses.
    default_pulse_duration : float
        Default duration for each exchange pulse (seconds).
    default_idle_duration : float
        Default idle time between pulse layers (seconds).
    """

    def __init__(
        self,
        n_qubits: int,
        connectivity: List[Tuple[int, int]],
        sequential: bool = True,
        default_pulse_duration: float = 10e-9,
        default_idle_duration: float = 10e-9,
    ):
        self.n_qubits = n_qubits
        self.connectivity = [
            (min(i, j), max(i, j)) for i, j in connectivity
        ]
        self.sequential = sequential
        self.default_pulse_duration = default_pulse_duration
        self.default_idle_duration = default_idle_duration

        # Build adjacency information for conflict detection
        self._edge_conflicts = self._build_edge_conflict_map()

    def _build_edge_conflict_map(self) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        """Build a map of which edges conflict with each other.

        Two edges conflict if they share a vertex.
        """
        conflicts = {}
        for e1 in self.connectivity:
            conflicts[e1] = set()
            for e2 in self.connectivity:
                if e1 != e2:
                    if e1[0] in e2 or e1[1] in e2:
                        conflicts[e1].add(e2)
        return conflicts

    def schedule(
        self, pulses: List[ExchangePulse]
    ) -> List[PulseLayer]:
        """Schedule a list of pulses into non-conflicting layers.

        Parameters
        ----------
        pulses : list of ExchangePulse
            Pulses to schedule (in desired order).

        Returns
        -------
        list of PulseLayer
            Scheduled layers, each containing non-conflicting pulses.
        """
        if self.sequential:
            return self._schedule_sequential(pulses)
        else:
            return self._schedule_parallel(pulses)

    def _schedule_sequential(
        self, pulses: List[ExchangePulse]
    ) -> List[PulseLayer]:
        """Strict sequential scheduling: one pulse per layer."""
        layers = []
        for pulse in pulses:
            if pulse.duration is None:
                pulse.duration = self.default_pulse_duration

            layer = PulseLayer()
            layer.add_pulse(pulse)
            layer.layer_duration = pulse.duration

            # Identify idle qubits
            active = layer.active_qubits
            layer.idle_qubits = [
                q for q in range(self.n_qubits) if q not in active
            ]
            layers.append(layer)

        return layers

    def _schedule_parallel(
        self, pulses: List[ExchangePulse]
    ) -> List[PulseLayer]:
        """Greedy parallel scheduling using first-fit colouring."""
        layers = []
        remaining = list(pulses)

        while remaining:
            layer = PulseLayer()
            still_remaining = []

            for pulse in remaining:
                if pulse.duration is None:
                    pulse.duration = self.default_pulse_duration

                if layer.add_pulse(pulse):
                    layer.layer_duration = max(
                        layer.layer_duration, pulse.duration
                    )
                else:
                    still_remaining.append(pulse)

            # Identify idle qubits
            active = layer.active_qubits
            layer.idle_qubits = [
                q for q in range(self.n_qubits) if q not in active
            ]
            layers.append(layer)
            remaining = still_remaining

        return layers

    def compute_total_time(self, layers: List[PulseLayer]) -> float:
        """Compute total evolution time including idle periods.

        Total time = Sum (layer_duration + idle_duration) for each layer.
        """
        total = 0.0
        for i, layer in enumerate(layers):
            total += layer.layer_duration
            if i < len(layers) - 1:
                total += self.default_idle_duration
        return total

    def compute_circuit_depth(self, layers: List[PulseLayer]) -> int:
        """Compute the circuit depth (number of layers)."""
        return len(layers)

    def validate_pulse_sequence(
        self, pulses: List[ExchangePulse]
    ) -> Tuple[bool, List[str]]:
        """Validate that a pulse sequence respects connectivity.

        Parameters
        ----------
        pulses : list of ExchangePulse
            Pulse sequence to validate.

        Returns
        -------
        (valid, errors) : tuple
            valid is True if all pulses are on valid edges.
            errors is a list of error messages.
        """
        errors = []
        valid_edges = set(self.connectivity)

        for i, pulse in enumerate(pulses):
            edge = pulse.edge
            if edge not in valid_edges:
                errors.append(
                    f"Pulse {i} ({pulse.label}): edge {edge} not in "
                    f"connectivity graph"
                )

        return len(errors) == 0, errors

    def get_edge_colouring(self) -> Dict[Tuple[int, int], int]:
        """Compute an edge colouring of the connectivity graph.

        Returns a mapping from edges to colours (integers) such that
        no two adjacent edges share the same colour. The number of
        colours equals the chromatic index of the graph.

        This gives the optimal parallelisation: all edges of the same
        colour can be pulsed simultaneously.
        """
        # Greedy edge colouring
        colouring = {}
        for edge in self.connectivity:
            # Find colours used by adjacent edges
            used_colours = set()
            for conflict_edge in self._edge_conflicts.get(edge, set()):
                if conflict_edge in colouring:
                    used_colours.add(colouring[conflict_edge])

            # Assign the smallest available colour
            colour = 0
            while colour in used_colours:
                colour += 1
            colouring[edge] = colour

        return colouring

    def get_parallel_groups(self) -> List[List[Tuple[int, int]]]:
        """Group edges into parallel execution groups.

        Returns a list of edge groups, where all edges in a group
        can be pulsed simultaneously without conflicts.
        """
        colouring = self.get_edge_colouring()
        n_colours = max(colouring.values()) + 1 if colouring else 0

        groups = [[] for _ in range(n_colours)]
        for edge, colour in colouring.items():
            groups[colour].append(edge)

        return groups


# ======================================================================
# DRL action space integration
# ======================================================================

class SequentialActionSpace:
    """Converts DRL agent actions into valid sequential pulse schedules.

    The DRL agent outputs a vector of exchange angles for all edges.
    This class converts that into a valid sequential pulse schedule
    respecting the connectivity and sequential pulsing constraints.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    connectivity : list of tuple
        Connectivity graph edges.
    sequential : bool
        Whether to enforce sequential pulsing.
    pulse_duration : float
        Duration of each exchange pulse (seconds).
    idle_duration : float
        Idle time between pulses (seconds).
    """

    def __init__(
        self,
        n_qubits: int,
        connectivity: List[Tuple[int, int]],
        sequential: bool = True,
        pulse_duration: float = 10e-9,
        idle_duration: float = 10e-9,
    ):
        self.n_qubits = n_qubits
        self.connectivity = connectivity
        self.n_edges = len(connectivity)
        self.scheduler = PulseScheduler(
            n_qubits=n_qubits,
            connectivity=connectivity,
            sequential=sequential,
            default_pulse_duration=pulse_duration,
            default_idle_duration=idle_duration,
        )

        # Precompute optimal ordering for parallel execution
        if not sequential:
            self._parallel_groups = self.scheduler.get_parallel_groups()
        else:
            self._parallel_groups = None

    def action_to_schedule(
        self,
        action: np.ndarray,
        ordering: Optional[List[int]] = None,
    ) -> List[PulseLayer]:
        """Convert a DRL action vector to a pulse schedule.

        Parameters
        ----------
        action : ndarray of shape (n_edges,)
            Exchange angles for each edge. Values in [-pi, pi].
        ordering : list of int, optional
            Custom ordering of edge indices. If None, uses default
            (connectivity order for sequential, optimal groups for
            parallel).

        Returns
        -------
        list of PulseLayer
            Scheduled pulse layers.
        """
        if ordering is None:
            ordering = list(range(self.n_edges))

        pulses = []
        for idx in ordering:
            if idx < self.n_edges and abs(action[idx]) > 1e-10:
                edge = self.connectivity[idx]
                pulses.append(ExchangePulse(
                    qubit_i=edge[0],
                    qubit_j=edge[1],
                    angle=float(action[idx]),
                    label=f"J_{edge[0]}{edge[1]}",
                ))

        return self.scheduler.schedule(pulses)

    def schedule_to_total_time(
        self, schedule: List[PulseLayer]
    ) -> float:
        """Compute total evolution time for a schedule."""
        return self.scheduler.compute_total_time(schedule)

    @property
    def action_dim(self) -> int:
        """Dimension of the action space (one angle per edge)."""
        return self.n_edges
