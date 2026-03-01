"""Tests for Monte Carlo simulation, crowd prediction, and pricing engine."""
import pytest
from datetime import datetime, date, time, timedelta
from uuid import uuid4

from trip_ai.core.models import (
    DayPlan,
    EventType,
    Itinerary,
    ItineraryStop,
    NodeType,
    RiskLevel,
    TravelerProfile,
    TravelNode,
)
from trip_ai.simulation import MonteCarloSimulator, CrowdPredictor, DynamicPricingEngine
from trip_ai.simulation.pricing_engine import PriceCategory


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def profile() -> TravelerProfile:
    return TravelerProfile(name="Test", budget_usd=1500, trip_duration_days=3)


@pytest.fixture
def node() -> TravelNode:
    return TravelNode(
        id="n1", name="Test Museum", node_type=NodeType.ATTRACTION,
        city="TestCity", country="TC", lat=35.0, lon=139.0,
        rating=4.5, avg_visit_duration_hours=2.0, entry_cost_usd=15,
        open_time=time(9, 0), close_time=time(18, 0),
        current_crowd_level=0.5, weather_comfort=0.9,
        tags=["culture", "museum"],
    )


@pytest.fixture
def simple_itinerary(profile, node) -> Itinerary:
    stop = ItineraryStop(
        node=node,
        arrival_time=datetime(2026, 3, 15, 10, 0),
        departure_time=datetime(2026, 3, 15, 12, 0),
    )
    day = DayPlan(day=1, date=datetime(2026, 3, 15), stops=[stop], estimated_cost_usd=15)
    return Itinerary(
        profile=profile,
        destination_city="TestCity",
        start_date=datetime(2026, 3, 15),
        days=[day],
        total_cost_usd=15,
    )


# ─── Monte Carlo ──────────────────────────────────────────────────────────────

class TestMonteCarloSimulator:

    def test_success_rate_in_range(self, simple_itinerary):
        sim = MonteCarloSimulator(n_iterations=1_000, seed=42)
        result = sim.simulate(simple_itinerary)
        assert 0.0 <= result.success_rate <= 1.0

    def test_avg_delay_non_negative(self, simple_itinerary):
        sim = MonteCarloSimulator(n_iterations=1_000, seed=42)
        result = sim.simulate(simple_itinerary)
        assert result.avg_delay_hours >= 0

    def test_worst_case_gte_avg(self, simple_itinerary):
        sim = MonteCarloSimulator(n_iterations=1_000, seed=42)
        result = sim.simulate(simple_itinerary)
        assert result.worst_case_delay_hours >= result.avg_delay_hours

    def test_risk_hotspots_returned(self, simple_itinerary):
        sim = MonteCarloSimulator(n_iterations=1_000, seed=42)
        result = sim.simulate(simple_itinerary)
        assert isinstance(result.risk_hotspots, list)

    def test_risk_level_classified(self, simple_itinerary):
        sim = MonteCarloSimulator(n_iterations=500, seed=42)
        result = sim.simulate(simple_itinerary)
        assert result.risk_level in list(RiskLevel)

    def test_more_iterations_stable(self, simple_itinerary):
        sim_small = MonteCarloSimulator(n_iterations=500, seed=1)
        sim_large = MonteCarloSimulator(n_iterations=5_000, seed=1)
        r_small = sim_small.simulate(simple_itinerary)
        r_large = sim_large.simulate(simple_itinerary)
        # Larger sample should converge — success rate within 10%
        assert abs(r_small.success_rate - r_large.success_rate) < 0.10

    def test_sensitivity_analysis(self, simple_itinerary):
        sim = MonteCarloSimulator(n_iterations=200, seed=42)
        sensitivity = sim.sensitivity_analysis(simple_itinerary, n_samples=100)
        assert isinstance(sensitivity, dict)
        assert len(sensitivity) > 0


# ─── Crowd Predictor ──────────────────────────────────────────────────────────

class TestCrowdPredictor:

    def test_predict_range(self, node):
        pred = CrowdPredictor(noise_std=0)
        at = datetime(2026, 7, 15, 12, 0)  # Summer, noon
        crowd = pred.predict(node, at)
        assert 0.0 <= crowd <= 1.0

    def test_weekend_busier_than_weekday(self, node):
        pred = CrowdPredictor(noise_std=0)
        weekday = datetime(2026, 3, 10, 12, 0)  # Tuesday
        weekend = datetime(2026, 3, 14, 12, 0)  # Saturday
        assert pred.predict(node, weekend) > pred.predict(node, weekday)

    def test_summer_busier_than_winter(self, node):
        pred = CrowdPredictor(noise_std=0)
        summer = datetime(2026, 8, 15, 12, 0)
        winter = datetime(2026, 1, 15, 12, 0)
        assert pred.predict(node, summer) > pred.predict(node, winter)

    def test_midday_busier_than_early_morning(self, node):
        pred = CrowdPredictor(noise_std=0)
        early = datetime(2026, 3, 15, 8, 0)
        midday = datetime(2026, 3, 15, 12, 0)
        assert pred.predict(node, midday) > pred.predict(node, early)

    def test_best_visit_time_in_open_hours(self, node):
        pred = CrowdPredictor(noise_std=0)
        best_h = pred.best_visit_time(node, datetime(2026, 3, 15), open_hour=9, close_hour=18)
        assert 9 <= best_h < 18

    def test_batch_predict(self, node):
        pred = CrowdPredictor()
        nodes = [node]
        result = pred.batch_predict(nodes, datetime(2026, 3, 15, 10, 0))
        assert node.id in result
        assert 0.0 <= result[node.id] <= 1.0

    def test_update_graph_crowds(self, node):
        pred = CrowdPredictor()
        updated = pred.update_graph_crowds([node], datetime(2026, 3, 15, 10, 0))
        assert len(updated) == 1
        assert 0.0 <= updated[0].current_crowd_level <= 1.0


# ─── Dynamic Pricing Engine ───────────────────────────────────────────────────

class TestDynamicPricingEngine:

    def test_predict_price_positive(self):
        eng = DynamicPricingEngine(noise_std=0)
        price = eng.predict_price(500, date(2026, 8, 15), PriceCategory.FLIGHT)
        assert price > 0

    def test_summer_flight_more_expensive(self):
        eng = DynamicPricingEngine(noise_std=0)
        summer = eng.predict_price(500, date(2026, 8, 15), PriceCategory.FLIGHT)
        winter = eng.predict_price(500, date(2026, 1, 15), PriceCategory.FLIGHT)
        assert summer > winter

    def test_price_calendar_length(self):
        eng = DynamicPricingEngine()
        cal = eng.price_calendar(200, PriceCategory.HOTEL, date.today(), days=14)
        assert len(cal) == 14

    def test_price_calendar_structure(self):
        eng = DynamicPricingEngine()
        cal = eng.price_calendar(200, PriceCategory.HOTEL, date.today(), days=5)
        for entry in cal:
            assert "date" in entry
            assert "predicted_price_usd" in entry
            assert entry["predicted_price_usd"] > 0

    def test_optimal_booking_window(self):
        eng = DynamicPricingEngine(noise_std=0)
        trip = date.today() + timedelta(days=60)
        result = eng.optimal_booking_window(600, trip, PriceCategory.FLIGHT)
        assert "best_booking_days_before" in result
        assert result["best_booking_days_before"] > 0

    def test_budget_forecast_total(self):
        eng = DynamicPricingEngine(noise_std=0)
        budget = eng.budget_forecast(
            flight_base=600,
            hotel_base_per_night=150,
            attraction_budget=200,
            trip_date=date(2026, 4, 10),
            n_nights=4,
        )
        assert budget["total_usd"] > 0
        expected = budget["flight_usd"] + budget["hotel_usd"] + budget["attractions_usd"] + \
                   budget["food_usd"] + budget["transport_usd"]
        assert abs(budget["total_usd"] - expected) < 1.0
