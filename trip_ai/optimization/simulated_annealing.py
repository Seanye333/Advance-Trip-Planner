"""
Simulated Annealing — City Route Optimizer
==========================================
Finds a near-optimal visit order for a fixed set of attractions
(i.e., solves TSP with cost = walking distance + travel time).

SA is chosen because:
  • TSP is NP-hard → SA finds near-optimal solutions quickly
  • Works well in practice for 5–25 stops
  • Easy to plug in multi-dimensional cost functions

Cooling schedule: exponential (T_k = T0 × α^k)
Neighbourhood:   2-opt swap (reverse a sub-sequence)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from trip_ai.graph_engine.travel_graph import TravelGraph, haversine_km
from trip_ai.core.models import TravelNode


@dataclass
class SAConfig:
    T0: float = 1000.0          # initial temperature
    T_min: float = 1e-3         # stopping temperature
    alpha: float = 0.995        # cooling rate
    max_iter: int = 50_000      # hard cap on iterations
    seed: int = 42


def _route_cost(
    route: list[TravelNode],
    graph: TravelGraph,
    w_distance: float = 0.5,
    w_time: float = 0.5,
) -> float:
    """Total cost of visiting attractions in given order."""
    if len(route) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(route, route[1:]):
        dist = haversine_km(a.lat, a.lon, b.lat, b.lon)
        edges = graph.get_edges(a.id, b.id)
        time_min = min((e.duration_minutes for e in edges), default=dist * 12)  # 5 km/h walking proxy
        total += w_distance * dist + w_time * (time_min / 60)
    return total


def _two_opt_swap(route: list[TravelNode], i: int, k: int) -> list[TravelNode]:
    """Reverse the segment between indices i and k (inclusive)."""
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]


class SimulatedAnnealingOptimizer:
    """
    SA-based TSP solver for ordering a list of TravelNodes.
    Returns the optimised route and its cost.
    """

    def __init__(self, config: SAConfig | None = None) -> None:
        self.config = config or SAConfig()

    def optimise(
        self,
        nodes: list[TravelNode],
        graph: TravelGraph,
    ) -> tuple[list[TravelNode], float]:
        """
        Returns (best_route, best_cost).
        The first and last element of best_route are the same node (round-trip)
        unless the list has fewer than 2 elements.
        """
        if len(nodes) <= 1:
            return nodes, 0.0

        rng = random.Random(self.config.seed)
        current = list(nodes)
        rng.shuffle(current)

        current_cost = _route_cost(current, graph)
        best = list(current)
        best_cost = current_cost

        T = self.config.T0
        iteration = 0

        while T > self.config.T_min and iteration < self.config.max_iter:
            # Generate a 2-opt neighbour
            i = rng.randint(0, len(current) - 2)
            k = rng.randint(i + 1, len(current) - 1)
            neighbour = _two_opt_swap(current, i, k)
            neighbour_cost = _route_cost(neighbour, graph)

            delta = neighbour_cost - current_cost
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current = neighbour
                current_cost = neighbour_cost
                if current_cost < best_cost:
                    best = list(current)
                    best_cost = current_cost

            T *= self.config.alpha
            iteration += 1

        return best, best_cost

    def optimise_multi_day(
        self,
        day_stops: list[list[TravelNode]],
        graph: TravelGraph,
    ) -> list[list[TravelNode]]:
        """Optimise route within each day independently."""
        return [self.optimise(stops, graph)[0] for stops in day_stops]
