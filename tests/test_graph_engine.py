"""Tests for the Travel Graph Engine."""
import json
import pytest
from datetime import time
from pathlib import Path

from trip_ai.graph_engine import TravelGraph
from trip_ai.core.models import NodeType, TransportMode, TravelEdge, TravelNode


@pytest.fixture
def sample_graph() -> TravelGraph:
    g = TravelGraph()
    data_path = Path(__file__).parent.parent / "data" / "sample_data.json"
    with open(data_path) as f:
        g.load_from_dict(json.load(f))
    return g


@pytest.fixture
def small_graph() -> TravelGraph:
    """Minimal hand-crafted graph for unit tests."""
    g = TravelGraph()
    hotel = TravelNode(
        id="hotel_a", name="Hotel A", node_type=NodeType.HOTEL,
        city="TestCity", country="TC", lat=35.0, lon=139.0,
        rating=4.0, avg_visit_duration_hours=0.0, entry_cost_usd=0,
        open_time=time(0, 0), close_time=time(23, 59),
    )
    museum = TravelNode(
        id="museum_b", name="Museum B", node_type=NodeType.ATTRACTION,
        city="TestCity", country="TC", lat=35.01, lon=139.01,
        rating=4.5, avg_visit_duration_hours=2.0, entry_cost_usd=10,
        open_time=time(9, 0), close_time=time(18, 0),
        tags=["culture", "history"],
    )
    park = TravelNode(
        id="park_c", name="Park C", node_type=NodeType.ATTRACTION,
        city="TestCity", country="TC", lat=35.02, lon=139.02,
        rating=4.2, avg_visit_duration_hours=1.5, entry_cost_usd=0,
        open_time=time(6, 0), close_time=time(20, 0),
        tags=["nature", "park"],
    )
    g.add_node(hotel)
    g.add_node(museum)
    g.add_node(park)
    g.add_edge(TravelEdge(
        source_id="hotel_a", target_id="museum_b",
        transport_mode=TransportMode.WALK,
        duration_minutes=15, cost_usd=0, reliability=0.99,
    ))
    g.add_edge(TravelEdge(
        source_id="museum_b", target_id="park_c",
        transport_mode=TransportMode.WALK,
        duration_minutes=10, cost_usd=0, reliability=0.99,
    ))
    return g


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestTravelGraph:

    def test_load_sample_data(self, sample_graph):
        assert sample_graph.node_count > 0
        assert sample_graph.edge_count > 0

    def test_nodes_in_city(self, sample_graph):
        tokyo_nodes = sample_graph.nodes_in_city("Tokyo")
        assert len(tokyo_nodes) > 0
        assert all(n.city == "Tokyo" for n in tokyo_nodes)

    def test_nodes_by_type(self, sample_graph):
        attractions = sample_graph.nodes_by_type(NodeType.ATTRACTION)
        assert all(n.node_type == NodeType.ATTRACTION for n in attractions)

    def test_add_and_get_node(self, small_graph):
        node = small_graph.get_node("museum_b")
        assert node is not None
        assert node.name == "Museum B"
        assert node.rating == 4.5

    def test_edges_from(self, small_graph):
        edges = small_graph.edges_from("hotel_a")
        assert len(edges) == 1
        assert edges[0].target_id == "museum_b"
        assert edges[0].duration_minutes == 15

    def test_shortest_path(self, small_graph):
        path = small_graph.shortest_path("hotel_a", "park_c", weight="duration_minutes")
        assert path == ["hotel_a", "museum_b", "park_c"]

    def test_shortest_path_no_route(self, small_graph):
        path = small_graph.shortest_path("park_c", "hotel_a")
        assert path == []  # directed graph — no back-edge

    def test_attractions_ranked(self, small_graph):
        ranked = small_graph.attractions_ranked("TestCity")
        assert len(ranked) == 2
        # Museum has higher rating → should rank first
        assert ranked[0].id == "museum_b"

    def test_attractions_ranked_by_tag(self, small_graph):
        ranked = small_graph.attractions_ranked("TestCity", profile_tags=["nature"])
        assert ranked[0].id == "park_c"  # nature tag matches

    def test_update_node_attr(self, small_graph):
        small_graph.update_node_attr("museum_b", current_crowd_level=0.9)
        node = small_graph.get_node("museum_b")
        assert node.current_crowd_level == 0.9

    def test_update_nonexistent_node(self, small_graph):
        with pytest.raises(KeyError):
            small_graph.update_node_attr("nonexistent", current_crowd_level=0.5)

    def test_walking_distance(self, small_graph):
        dist = small_graph.walking_tour_distance_km(["hotel_a", "museum_b", "park_c"])
        assert dist > 0

    def test_json_roundtrip(self, small_graph):
        json_str = small_graph.to_json()
        restored = TravelGraph.from_json(json_str)
        assert restored.node_count == small_graph.node_count
        assert restored.edge_count == small_graph.edge_count

    def test_city_subgraph(self, sample_graph):
        sub = sample_graph.city_subgraph("Tokyo")
        assert sub.node_count <= sample_graph.node_count
        for _, data in sub._g.nodes(data=True):
            assert data.get("city") == "Tokyo"
