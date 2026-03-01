"""
Hybrid Planner — LLM + Optimization
======================================
Orchestrates the full planning pipeline:

  1. LLM drafts a high-level plan (reasoning)
  2. NSGA-II optimises attraction selection (math)
  3. Simulated Annealing optimises daily route order (TSP)
  4. LLM writes the final narrative explanation

This is the main public interface for end-to-end trip generation.
"""
from __future__ import annotations

from datetime import datetime

from trip_ai.ai_planner.llm_planner import LLMPlanner
from trip_ai.core.models import (
    Itinerary,
    OptimizationResult,
    OptimizationWeights,
    TravelerProfile,
    TravelNode,
)
from trip_ai.graph_engine.travel_graph import TravelGraph
from trip_ai.optimization.multi_objective import TripOptimizer
from trip_ai.optimization.simulated_annealing import SimulatedAnnealingOptimizer


class HybridTripPlanner:
    """
    End-to-end trip planner that combines LLM reasoning with
    mathematical optimisation for the best of both worlds.
    """

    def __init__(
        self,
        graph: TravelGraph,
        llm: LLMPlanner | None = None,
        optimizer: TripOptimizer | None = None,
        sa_optimizer: SimulatedAnnealingOptimizer | None = None,
    ) -> None:
        self.graph = graph
        self.llm = llm or LLMPlanner()
        self.optimizer = optimizer or TripOptimizer()
        self.sa = sa_optimizer or SimulatedAnnealingOptimizer()

    def plan(
        self,
        profile: TravelerProfile,
        city: str,
        start_date: datetime | None = None,
        weights: OptimizationWeights | None = None,
        max_candidates: int = 25,
        use_llm_draft: bool = True,
    ) -> OptimizationResult:
        """
        Full pipeline: gather candidates → [LLM draft] → NSGA-II → SA → narrative.

        Parameters
        ----------
        profile        : traveller preferences and constraints
        city           : destination city name
        start_date     : trip start (defaults to now)
        weights        : trade-off weights for multi-objective optimisation
        max_candidates : cap on attractions fed to optimizer (performance)
        use_llm_draft  : whether to use Claude for initial reasoning
        """
        start_date = start_date or datetime.utcnow()

        # ── Step 1: Gather candidate attractions ──────────────────────────
        preference_tags = self._profile_to_tags(profile)
        candidates = self.graph.attractions_ranked(
            city=city,
            profile_tags=preference_tags,
            max_crowd=1.0 - profile.crowd_tolerance * 0.3,
        )[:max_candidates]

        if not candidates:
            raise ValueError(f"No attractions found for city '{city}'")

        # ── Step 2: LLM Draft (optional reasoning pass) ───────────────────
        llm_draft = ""
        if use_llm_draft and candidates:
            try:
                llm_draft = self.llm.draft_itinerary(profile, city, candidates)
            except Exception as e:
                llm_draft = f"[LLM unavailable: {e}]"

        # ── Step 3: Build travel time vector (hotel → each attraction) ────
        travel_times = self._estimate_travel_times(candidates)

        # ── Step 4: NSGA-II multi-objective optimisation ──────────────────
        result: OptimizationResult = self.optimizer.optimize(
            candidates=candidates,
            profile=profile,
            travel_times_minutes=travel_times,
            weights=weights,
            start_date=start_date,
        )

        # ── Step 5: SA route optimisation within each day ─────────────────
        day_stops = [[stop.node for stop in day.stops] for day in result.itinerary.days]
        optimised_day_stops = self.sa.optimise_multi_day(day_stops, self.graph)

        # Reorder stops in each day according to SA result
        for day, optimised_stops in zip(result.itinerary.days, optimised_day_stops):
            stop_map = {s.node.id: s for s in day.stops}
            day.stops = [stop_map[n.id] for n in optimised_stops if n.id in stop_map]

        # ── Step 6: LLM narrative explanation ─────────────────────────────
        try:
            narrative = self.llm.explain_itinerary(result.itinerary)
        except Exception as e:
            narrative = f"Optimised {profile.trip_duration_days}-day trip to {city}. [LLM: {e}]"

        result.itinerary.ai_narrative = narrative
        return result

    def _profile_to_tags(self, profile: TravelerProfile) -> list[str]:
        """Convert preference scores into tag list for graph query."""
        tags = []
        if profile.pref_culture > 0.5:
            tags.extend(["culture", "history", "art", "museum"])
        if profile.pref_nature > 0.5:
            tags.extend(["nature", "park", "wildlife"])
        if profile.pref_food > 0.5:
            tags.extend(["food", "market", "restaurant"])
        if profile.pref_adventure > 0.5:
            tags.extend(["adventure", "sport"])
        if profile.pref_relaxation > 0.5:
            tags.extend(["relaxation", "beach", "spa"])
        return tags

    def _estimate_travel_times(self, candidates: list[TravelNode]) -> list[float]:
        """
        Rough travel time from a central hotel to each attraction.
        In production, query a real routing API (Mapbox, Google Maps).
        Proxy: 5 km/h walking speed based on distance from city centroid.
        """
        if not candidates:
            return []
        # City centroid
        lat_c = sum(n.lat for n in candidates) / len(candidates)
        lon_c = sum(n.lon for n in candidates) / len(candidates)

        from trip_ai.graph_engine.travel_graph import haversine_km
        times = []
        for node in candidates:
            dist_km = haversine_km(lat_c, lon_c, node.lat, node.lon)
            walk_min = dist_km / 5 * 60  # 5 km/h
            times.append(walk_min)
        return times
