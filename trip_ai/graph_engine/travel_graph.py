"""
Travel Graph Engine
===================
Models the travel world as a directed multigraph where:
  • Nodes  = places (airports, hotels, attractions, restaurants, transport hubs)
  • Edges  = connections (flights, trains, walking, driving, …)

Supports:
  - Dynamic attribute updates (crowd, weather, closures)
  - Haversine distance calculation
  - Subgraph extraction for a single city
  - JSON import/export for persistence
"""
from __future__ import annotations

import json
import math
from typing import Any

import networkx as nx

from trip_ai.core.models import NodeType, TransportMode, TravelEdge, TravelNode


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two GPS points in km."""
    R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class TravelGraph:
    """
    Core directed multigraph of all nodes and transport edges.

    Each edge can carry multiple transport modes between the same pair of nodes
    (e.g. both 'train' and 'car' between City A and City B).
    """

    def __init__(self) -> None:
        # DiGraph allows directed edges (A→B may differ from B→A)
        # Multigraph lets us store multiple edges (different transport modes)
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()

    # ─── Node management ──────────────────────────────────────────────────────

    def add_node(self, node: TravelNode) -> None:
        self._g.add_node(node.id, **node.model_dump())

    def get_node(self, node_id: str) -> TravelNode | None:
        data = self._g.nodes.get(node_id)
        return TravelNode(**data) if data else None

    def update_node_attr(self, node_id: str, **kwargs: Any) -> None:
        """Update live attributes (crowd, weather, closure status, …)."""
        if node_id not in self._g:
            raise KeyError(f"Node '{node_id}' not found in graph")
        self._g.nodes[node_id].update(kwargs)

    def nodes_by_type(self, node_type: NodeType) -> list[TravelNode]:
        return [
            TravelNode(**d)
            for _, d in self._g.nodes(data=True)
            if d.get("node_type") == node_type.value
        ]

    def nodes_in_city(self, city: str) -> list[TravelNode]:
        return [
            TravelNode(**d)
            for _, d in self._g.nodes(data=True)
            if d.get("city", "").lower() == city.lower()
        ]

    # ─── Edge management ──────────────────────────────────────────────────────

    def add_edge(self, edge: TravelEdge) -> None:
        self._g.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.transport_mode.value,
            **edge.model_dump(),
        )

    def get_edges(
        self,
        source_id: str,
        target_id: str,
    ) -> list[TravelEdge]:
        edges = self._g.get_edge_data(source_id, target_id) or {}
        return [TravelEdge(**data) for data in edges.values()]

    def edges_from(self, node_id: str) -> list[TravelEdge]:
        return [
            TravelEdge(**data)
            for _, _, data in self._g.out_edges(node_id, data=True)
        ]

    # ─── Graph queries ────────────────────────────────────────────────────────

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        weight: str = "duration_minutes",
    ) -> list[str]:
        """Dijkstra shortest path by a single edge weight."""
        try:
            return nx.dijkstra_path(self._g, source_id, target_id, weight=weight)
        except nx.NetworkXNoPath:
            return []

    def city_subgraph(self, city: str) -> "TravelGraph":
        """Return a new TravelGraph containing only nodes in *city*."""
        node_ids = [
            n
            for n, d in self._g.nodes(data=True)
            if d.get("city", "").lower() == city.lower()
        ]
        sub = TravelGraph()
        sub._g = self._g.subgraph(node_ids).copy()
        return sub

    def attractions_ranked(
        self,
        city: str,
        profile_tags: list[str] | None = None,
        max_crowd: float = 1.0,
    ) -> list[TravelNode]:
        """
        Return attractions in *city* sorted by a composite score:
            score = rating × weather_comfort × (1 − crowd_penalty)
        Optionally filter by tag overlap with *profile_tags*.
        """
        attractions = [
            n
            for n in self.nodes_in_city(city)
            if n.node_type == NodeType.ATTRACTION
            and n.current_crowd_level <= max_crowd
        ]
        if profile_tags:
            tag_set = set(t.lower() for t in profile_tags)
            attractions = [
                n for n in attractions
                if tag_set & set(t.lower() for t in n.tags)
            ] or attractions  # fall back to all if none match

        def score(n: TravelNode) -> float:
            crowd_penalty = n.current_crowd_level * 0.3
            return n.rating * n.weather_comfort * (1 - crowd_penalty)

        return sorted(attractions, key=score, reverse=True)

    def walking_tour_distance_km(self, node_ids: list[str]) -> float:
        """Estimate total walking distance for an ordered list of stops."""
        total = 0.0
        nodes = [self.get_node(nid) for nid in node_ids]
        for a, b in zip(nodes, nodes[1:]):
            if a and b:
                total += haversine_km(a.lat, a.lon, b.lat, b.lon)
        return total

    # ─── Persistence ──────────────────────────────────────────────────────────

    def to_json(self) -> str:
        data = nx.node_link_data(self._g)
        return json.dumps(data, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "TravelGraph":
        g = cls()
        data = json.loads(json_str)
        g._g = nx.node_link_graph(data, directed=True, multigraph=True)
        return g

    def load_from_dict(self, payload: dict) -> None:
        """
        Load nodes + edges from a structured dict (e.g. parsed from JSON file).

        Expected format:
        {
          "nodes": [ { <TravelNode fields> }, … ],
          "edges": [ { <TravelEdge fields> }, … ]
        }
        """
        for n in payload.get("nodes", []):
            self.add_node(TravelNode(**n))
        for e in payload.get("edges", []):
            self.add_edge(TravelEdge(**e))

    # ─── Stats ────────────────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return self._g.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._g.number_of_edges()

    def __repr__(self) -> str:
        return f"TravelGraph(nodes={self.node_count}, edges={self.edge_count})"
