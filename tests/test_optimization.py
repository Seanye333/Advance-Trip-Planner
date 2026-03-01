"""Tests for the Optimization Engine."""
import pytest
from datetime import datetime, time

from trip_ai.core.models import (
    NodeType,
    OptimizationWeights,
    TravelerProfile,
    TravelNode,
)
from trip_ai.graph_engine import TravelGraph
from trip_ai.optimization.multi_objective import (
    TripOptimizer,
    experience_score,
    fatigue_score,
)
from trip_ai.optimization.simulated_annealing import (
    SimulatedAnnealingOptimizer,
    SAConfig,
)


@pytest.fixture
def profile() -> TravelerProfile:
    return TravelerProfile(
        name="Tester",
        budget_usd=800,
        trip_duration_days=3,
        pref_culture=0.8,
        pref_nature=0.5,
        pref_food=0.6,
        pref_adventure=0.3,
    )


@pytest.fixture
def attractions() -> list[TravelNode]:
    def make(id_, name, cost, duration, rating, tags, crowd=0.5):
        return TravelNode(
            id=id_, name=name, node_type=NodeType.ATTRACTION,
            city="TestCity", country="TC",
            lat=35.0 + len(id_) * 0.01, lon=139.0 + len(id_) * 0.01,
            rating=rating, avg_visit_duration_hours=duration,
            entry_cost_usd=cost, open_time=time(9, 0), close_time=time(18, 0),
            current_crowd_level=crowd, weather_comfort=0.9, tags=tags,
        )

    return [
        make("a1", "Museum of Art",     cost=10,  duration=2.0, rating=4.7, tags=["culture", "art", "museum"]),
        make("a2", "History Museum",    cost=8,   duration=1.5, rating=4.5, tags=["culture", "history"]),
        make("a3", "National Park",     cost=5,   duration=3.0, rating=4.6, tags=["nature", "park"]),
        make("a4", "Food Market",       cost=0,   duration=1.0, rating=4.3, tags=["food", "market"]),
        make("a5", "Adventure Park",    cost=25,  duration=2.5, rating=4.2, tags=["adventure", "sport"]),
        make("a6", "Botanical Garden",  cost=6,   duration=2.0, rating=4.4, tags=["nature", "park", "relaxation"]),
        make("a7", "Science Museum",    cost=12,  duration=2.0, rating=4.5, tags=["culture", "museum", "indoor"]),
        make("a8", "Street Food Tour",  cost=30,  duration=2.5, rating=4.8, tags=["food", "culture", "adventure"]),
    ]


# ─── Experience / Fatigue helpers ─────────────────────────────────────────────

class TestHelpers:

    def test_experience_score_empty(self, profile):
        assert experience_score([], profile) == 0.0

    def test_experience_score_range(self, attractions, profile):
        score = experience_score(attractions, profile)
        assert 0.0 <= score <= 1.0

    def test_experience_score_prefers_culture(self, attractions, profile):
        culture_only = [a for a in attractions if "culture" in a.tags]
        nature_only  = [a for a in attractions if "nature" in a.tags and "culture" not in a.tags]
        # High culture preference → culture nodes score higher
        assert experience_score(culture_only, profile) >= experience_score(nature_only, profile)

    def test_fatigue_score_range(self, attractions):
        assert 0.0 <= fatigue_score(attractions) <= 1.0

    def test_fatigue_score_empty(self):
        assert fatigue_score([]) == 0.0

    def test_fatigue_increases_with_more_stops(self, attractions):
        f_few  = fatigue_score(attractions[:2])
        f_many = fatigue_score(attractions)
        assert f_many >= f_few


# ─── NSGA-II Optimizer ────────────────────────────────────────────────────────

class TestTripOptimizer:

    def test_optimize_returns_result(self, attractions, profile):
        optimizer = TripOptimizer(pop_size=20, n_gen=10)
        travel_times = [15.0] * len(attractions)
        result = optimizer.optimize(
            candidates=attractions,
            profile=profile,
            travel_times_minutes=travel_times,
            start_date=datetime(2026, 3, 15),
        )
        assert result.itinerary is not None
        assert result.optimization_time_seconds > 0
        assert result.algorithm == "NSGA-II"

    def test_budget_constraint_respected(self, attractions, profile):
        optimizer = TripOptimizer(pop_size=20, n_gen=10)
        travel_times = [10.0] * len(attractions)
        result = optimizer.optimize(
            candidates=attractions,
            profile=profile,
            travel_times_minutes=travel_times,
        )
        assert result.itinerary.total_cost_usd <= profile.budget_usd * 1.05  # 5% tolerance

    def test_itinerary_has_days(self, attractions, profile):
        optimizer = TripOptimizer(pop_size=20, n_gen=10)
        travel_times = [10.0] * len(attractions)
        result = optimizer.optimize(
            candidates=attractions,
            profile=profile,
            travel_times_minutes=travel_times,
            start_date=datetime(2026, 3, 15),
        )
        assert len(result.itinerary.days) >= 1

    def test_custom_weights(self, attractions, profile):
        optimizer = TripOptimizer(pop_size=20, n_gen=10)
        travel_times = [10.0] * len(attractions)
        weights = OptimizationWeights(experience=0.1, cost=0.7, time=0.1, crowd=0.05, fatigue=0.05)
        result = optimizer.optimize(
            candidates=attractions,
            profile=profile,
            travel_times_minutes=travel_times,
            weights=weights,
        )
        assert result.itinerary is not None

    def test_pareto_solutions_recorded(self, attractions, profile):
        optimizer = TripOptimizer(pop_size=30, n_gen=15)
        travel_times = [10.0] * len(attractions)
        result = optimizer.optimize(
            candidates=attractions,
            profile=profile,
            travel_times_minutes=travel_times,
        )
        assert len(result.pareto_solutions) > 0


# ─── Simulated Annealing ──────────────────────────────────────────────────────

class TestSimulatedAnnealing:

    def test_optimise_single_node(self, attractions):
        g = TravelGraph()
        sa = SimulatedAnnealingOptimizer(SAConfig(max_iter=100))
        route, cost = sa.optimise([attractions[0]], g)
        assert len(route) == 1
        assert cost == 0.0

    def test_optimise_returns_same_nodes(self, attractions):
        g = TravelGraph()
        sa = SimulatedAnnealingOptimizer(SAConfig(max_iter=500, seed=42))
        route, cost = sa.optimise(attractions[:4], g)
        assert set(n.id for n in route) == set(n.id for n in attractions[:4])

    def test_optimise_multi_day(self, attractions):
        g = TravelGraph()
        sa = SimulatedAnnealingOptimizer(SAConfig(max_iter=200))
        day_groups = [attractions[:3], attractions[3:6], attractions[6:]]
        result = sa.optimise_multi_day(day_groups, g)
        assert len(result) == 3

    def test_different_seeds_may_differ(self, attractions):
        g = TravelGraph()
        sa1 = SimulatedAnnealingOptimizer(SAConfig(max_iter=500, seed=1))
        sa2 = SimulatedAnnealingOptimizer(SAConfig(max_iter=500, seed=99))
        r1, c1 = sa1.optimise(attractions[:5], g)
        r2, c2 = sa2.optimise(attractions[:5], g)
        # Both routes have all nodes — costs may differ
        assert set(n.id for n in r1) == set(n.id for n in r2)
