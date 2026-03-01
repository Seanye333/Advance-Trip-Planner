"""Tests for the Real-Time Replanning Engine."""
import pytest
from datetime import datetime, time, timedelta

from trip_ai.core.models import (
    DayPlan,
    EventType,
    Itinerary,
    ItineraryStop,
    NodeType,
    TravelerProfile,
    TravelNode,
)
from trip_ai.graph_engine import TravelGraph
from trip_ai.replanning import ReplanEngine, DisruptionEventHandler


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def profile():
    return TravelerProfile(name="Tester", budget_usd=1000, trip_duration_days=3)


def make_node(id_, name, tags=None, cost=10.0, crowd=0.5):
    return TravelNode(
        id=id_, name=name, node_type=NodeType.ATTRACTION,
        city="TestCity", country="TC", lat=35.0, lon=139.0,
        rating=4.5, avg_visit_duration_hours=1.5,
        entry_cost_usd=cost, open_time=time(9, 0), close_time=time(18, 0),
        current_crowd_level=crowd, weather_comfort=0.8,
        tags=tags or ["culture"],
    )


def make_stop(node, hour_arrive=10, hour_depart=12):
    d = datetime(2026, 3, 15)
    return ItineraryStop(
        node=node,
        arrival_time=d.replace(hour=hour_arrive),
        departure_time=d.replace(hour=hour_depart),
    )


@pytest.fixture
def multi_day_itinerary(profile):
    nodes = [
        make_node("n1", "Museum", tags=["culture", "museum"], cost=15),
        make_node("n2", "Park", tags=["nature", "park"], cost=5),
        make_node("n3", "Beach", tags=["nature", "beach"], cost=0),
        make_node("n4", "Gallery", tags=["art", "culture"], cost=12),
        make_node("n5", "Market", tags=["food", "market"], cost=0),
    ]
    day1 = DayPlan(
        day=1, date=datetime(2026, 3, 15),
        stops=[make_stop(nodes[0], 10, 12), make_stop(nodes[1], 13, 14)],
        estimated_cost_usd=20,
    )
    day2 = DayPlan(
        day=2, date=datetime(2026, 3, 16),
        stops=[make_stop(nodes[2], 10, 12), make_stop(nodes[3], 13, 15)],
        estimated_cost_usd=12,
    )
    day3 = DayPlan(
        day=3, date=datetime(2026, 3, 17),
        stops=[make_stop(nodes[4], 10, 11)],
        estimated_cost_usd=0,
    )
    return Itinerary(
        profile=profile,
        destination_city="TestCity",
        start_date=datetime(2026, 3, 15),
        days=[day1, day2, day3],
        total_cost_usd=32,
    )


@pytest.fixture
def graph_with_alternatives():
    g = TravelGraph()
    indoor_alt = make_node("indoor_1", "Indoor Gallery", tags=["art", "culture", "museum", "indoor"], cost=8)
    park_outdoor = make_node("park_out", "Outdoor Park", tags=["nature", "park"], cost=0)
    g.add_node(indoor_alt)
    g.add_node(park_outdoor)
    return g


@pytest.fixture
def replan_engine(graph_with_alternatives):
    return ReplanEngine(graph_with_alternatives)


@pytest.fixture
def event_handler():
    return DisruptionEventHandler()


# ─── DisruptionEventHandler ───────────────────────────────────────────────────

class TestDisruptionEventHandler:

    def test_create_flight_delay(self, event_handler):
        ev = event_handler.create_flight_delay(delay_hours=3.5, flight_node_id="airport_x")
        assert ev.event_type == EventType.FLIGHT_DELAY
        assert ev.severity == pytest.approx(3.5 / 12, rel=0.01)
        assert ev.details["delay_hours"] == 3.5

    def test_create_closure(self, event_handler):
        ev = event_handler.create_closure("museum_1", reason="renovation")
        assert ev.event_type == EventType.CLOSURE
        assert ev.affected_node_id == "museum_1"

    def test_create_weather_event(self, event_handler):
        ev = event_handler.create_weather_event(severity=0.8, affected_city="Tokyo", weather_type="typhoon")
        assert ev.event_type == EventType.WEATHER_CHANGE
        assert ev.severity == 0.8

    def test_create_fatigue_event(self, event_handler):
        ev = event_handler.create_fatigue_event(current_energy=0.2)
        assert ev.event_type == EventType.FATIGUE
        assert ev.severity == pytest.approx(0.8, rel=0.01)

    def test_requires_full_replan_strike(self, event_handler):
        ev = event_handler.create_closure("n1", reason="strike")
        ev.event_type = EventType.STRIKE
        assert event_handler.requires_full_replan(ev) is True

    def test_requires_full_replan_small_delay(self, event_handler):
        ev = event_handler.create_flight_delay(0.5, "airport")
        assert event_handler.requires_full_replan(ev) is False


# ─── ReplanEngine ─────────────────────────────────────────────────────────────

class TestReplanEngine:

    def test_flight_delay_shifts_times(self, replan_engine, multi_day_itinerary, event_handler):
        original_arrival = multi_day_itinerary.days[0].stops[0].arrival_time
        ev = event_handler.create_flight_delay(2.0, "airport")
        result = replan_engine.replan(multi_day_itinerary, ev, current_day=1)
        new_arrival = result.new_itinerary.days[0].stops[0].arrival_time
        assert new_arrival == original_arrival + timedelta(hours=2)

    def test_flight_delay_returns_changes(self, replan_engine, multi_day_itinerary, event_handler):
        ev = event_handler.create_flight_delay(2.0, "airport")
        result = replan_engine.replan(multi_day_itinerary, ev, current_day=1)
        assert len(result.changes_summary) >= 1

    def test_long_delay_drops_last_stop(self, replan_engine, multi_day_itinerary, event_handler):
        original_stops_day1 = len(multi_day_itinerary.days[0].stops)
        ev = event_handler.create_flight_delay(5.0, "airport")  # >4h → drop stop
        result = replan_engine.replan(multi_day_itinerary, ev, current_day=1)
        new_stops_day1 = len(result.new_itinerary.days[0].stops)
        assert new_stops_day1 < original_stops_day1

    def test_closure_removes_node(self, replan_engine, multi_day_itinerary, event_handler):
        target_id = multi_day_itinerary.days[0].stops[0].node.id
        ev = event_handler.create_closure(target_id)
        result = replan_engine.replan(multi_day_itinerary, ev, current_day=1)
        all_stop_ids = [
            s.node.id
            for d in result.new_itinerary.days
            for s in d.stops
        ]
        assert target_id not in all_stop_ids

    def test_fatigue_removes_stops(self, replan_engine, multi_day_itinerary, event_handler):
        original_count = len(multi_day_itinerary.days[0].stops)
        ev = event_handler.create_fatigue_event(current_energy=0.1)  # very tired
        result = replan_engine.replan(multi_day_itinerary, ev, current_day=1)
        new_count = len(result.new_itinerary.days[0].stops)
        assert new_count <= original_count

    def test_traffic_adds_buffer(self, replan_engine, multi_day_itinerary):
        from trip_ai.core.models import DisruptionEvent
        original_arrival = multi_day_itinerary.days[0].stops[0].arrival_time
        ev = DisruptionEvent(event_type=EventType.TRAFFIC_SPIKE, severity=0.8)
        result = replan_engine.replan(multi_day_itinerary, ev, current_day=1)
        new_arrival = result.new_itinerary.days[0].stops[0].arrival_time
        assert new_arrival >= original_arrival

    def test_replan_time_ms_recorded(self, replan_engine, multi_day_itinerary, event_handler):
        ev = event_handler.create_flight_delay(1.0, "airport")
        result = replan_engine.replan(multi_day_itinerary, ev)
        assert result.replan_time_ms > 0

    def test_original_itinerary_unchanged(self, replan_engine, multi_day_itinerary, event_handler):
        original_stops = [s.node.id for s in multi_day_itinerary.days[0].stops]
        ev = event_handler.create_flight_delay(5.0, "airport")
        replan_engine.replan(multi_day_itinerary, ev)
        # Original itinerary should be deep-copied, not mutated
        assert [s.node.id for s in multi_day_itinerary.days[0].stops] == original_stops
