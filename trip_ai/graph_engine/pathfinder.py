"""
Multi-Objective Pathfinder
==========================
Finds Pareto-optimal paths through the travel graph balancing:
  - Travel time
  - Cost
  - Reliability (inverse of risk)
  - Comfort (inverse of crowd + fatigue)

Uses a label-setting algorithm that tracks multiple objective vectors
simultaneously, returning the full Pareto frontier of paths.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Iterator

from trip_ai.core.models import TransportMode, TravelEdge
from trip_ai.graph_engine.travel_graph import TravelGraph


@dataclass(order=True)
class PathLabel:
    """A partial path and its accumulated objective costs."""
    # Objectives (minimised)
    cost_time: float
    cost_money: float
    cost_risk: float
    # Path tracking (not used for ordering)
    node_id: str = field(compare=False)
    path: list[str] = field(default_factory=list, compare=False)
    edges: list[TravelEdge] = field(default_factory=list, compare=False)

    def dominates(self, other: "PathLabel") -> bool:
        """True if *self* is at least as good on all objectives and strictly better on one."""
        return (
            self.cost_time  <= other.cost_time  and
            self.cost_money <= other.cost_money and
            self.cost_risk  <= other.cost_risk  and
            (
                self.cost_time  < other.cost_time  or
                self.cost_money < other.cost_money or
                self.cost_risk  < other.cost_risk
            )
        )

    @property
    def weighted_cost(self) -> float:
        """Scalar proxy for heap ordering (equal weights)."""
        return self.cost_time + self.cost_money + self.cost_risk


class MultiObjectivePathfinder:
    """
    Label-setting multi-objective shortest path on the TravelGraph.

    For larger graphs, consider A* with a dominance check or
    replace with NSGA-II over candidate path sets.
    """

    def __init__(self, graph: TravelGraph) -> None:
        self._graph = graph

    def find_pareto_paths(
        self,
        source_id: str,
        target_id: str,
        allowed_modes: list[TransportMode] | None = None,
        max_paths: int = 10,
    ) -> list[PathLabel]:
        """
        Return the Pareto-optimal set of paths from *source_id* to *target_id*.

        Parameters
        ----------
        allowed_modes : restrict which transport modes may be used
        max_paths     : cap on returned solutions (first-found heuristic)
        """
        pareto: list[PathLabel] = []  # final non-dominated solutions
        # { node_id: [labels reaching it that aren't yet dominated] }
        labels: dict[str, list[PathLabel]] = {source_id: []}

        start = PathLabel(
            cost_time=0.0,
            cost_money=0.0,
            cost_risk=0.0,
            node_id=source_id,
            path=[source_id],
        )

        # Min-heap on weighted_cost for greedy expansion
        heap: list[tuple[float, PathLabel]] = [(0.0, start)]

        while heap and len(pareto) < max_paths:
            _, current = heapq.heappop(heap)

            if current.node_id == target_id:
                # Check if it's non-dominated vs existing solutions
                if not any(p.dominates(current) for p in pareto):
                    pareto = [p for p in pareto if not current.dominates(p)]
                    pareto.append(current)
                continue

            for edge in self._graph.edges_from(current.node_id):
                if allowed_modes and edge.transport_mode not in allowed_modes:
                    continue
                next_id = edge.target_id
                if next_id in current.path:
                    continue  # no cycles

                risk_cost = (1 - edge.reliability) * edge.duration_minutes  # penalise unreliable legs

                nxt = PathLabel(
                    cost_time=current.cost_time + edge.duration_minutes,
                    cost_money=current.cost_money + edge.cost_usd,
                    cost_risk=current.cost_risk + risk_cost,
                    node_id=next_id,
                    path=current.path + [next_id],
                    edges=current.edges + [edge],
                )

                existing = labels.get(next_id, [])
                if any(lbl.dominates(nxt) for lbl in existing):
                    continue  # dominated — skip

                # Prune dominated existing labels
                labels[next_id] = [l for l in existing if not nxt.dominates(l)]
                labels[next_id].append(nxt)
                heapq.heappush(heap, (nxt.weighted_cost, nxt))

        return pareto

    def best_path(
        self,
        source_id: str,
        target_id: str,
        weights: dict[str, float] | None = None,
        allowed_modes: list[TransportMode] | None = None,
    ) -> PathLabel | None:
        """
        Return a single best path using a weighted-sum scalarisation of the Pareto front.

        weights: {"time": 0.5, "money": 0.3, "risk": 0.2}
        """
        w = weights or {"time": 0.4, "money": 0.4, "risk": 0.2}
        paths = self.find_pareto_paths(source_id, target_id, allowed_modes)
        if not paths:
            return None

        def scalar(p: PathLabel) -> float:
            return (
                w.get("time", 0)  * p.cost_time +
                w.get("money", 0) * p.cost_money +
                w.get("risk", 0)  * p.cost_risk
            )

        return min(paths, key=scalar)

    def tsp_nearest_neighbor(self, node_ids: list[str]) -> list[str]:
        """
        Greedy nearest-neighbour heuristic for Travelling Salesman Problem.
        Returns an ordered visit sequence minimising total travel time.
        Used as the initial solution seed for simulated annealing.
        """
        if not node_ids:
            return []

        unvisited = set(node_ids)
        route = [node_ids[0]]
        unvisited.discard(node_ids[0])

        while unvisited:
            current = route[-1]
            # Find the closest unvisited node by shortest path time
            best_next, best_time = None, float("inf")
            for candidate in unvisited:
                edges = self._graph.edges_from(current)
                direct = next(
                    (e for e in edges if e.target_id == candidate), None
                )
                t = direct.duration_minutes if direct else float("inf")
                if t < best_time:
                    best_time = t
                    best_next = candidate

            if best_next is None:
                break
            route.append(best_next)
            unvisited.discard(best_next)

        return route
