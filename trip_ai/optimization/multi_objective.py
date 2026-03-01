"""
Multi-Objective Trip Optimizer — NSGA-II
=========================================
Uses pymoo's NSGA-II algorithm to find the Pareto-optimal set of itineraries
that simultaneously optimise:

  f1 = -experience_score  (maximise → minimise negative)
  f2 =  total_cost_usd    (minimise)
  f3 =  total_time_hours  (minimise)
  f4 =  avg_crowd_level   (minimise)
  f5 =  total_fatigue     (minimise)

Decision variables:
  x[i] ∈ {0, 1}  — whether attraction i is included
  (continuous relaxation during optimisation, rounded at the end)

After NSGA-II converges, we select the solution that best matches the
traveller's OptimizationWeights using a weighted Tchebycheff scalarisation.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from trip_ai.core.models import (
    DayPlan,
    Itinerary,
    ItineraryStop,
    OptimizationResult,
    OptimizationWeights,
    TravelerProfile,
    TravelNode,
)


# ─── Objective helper ─────────────────────────────────────────────────────────

def fatigue_score(attractions: list[TravelNode]) -> float:
    """
    Simple fatigue model: each attraction adds a base load.
    Long visits and high-crowd nodes are more tiring.
    """
    fatigue = 0.0
    for n in attractions:
        base = n.avg_visit_duration_hours * 0.1
        crowd_load = n.current_crowd_level * 0.15
        fatigue += base + crowd_load
    return min(fatigue, 1.0)


def experience_score(
    attractions: list[TravelNode],
    profile: TravelerProfile,
) -> float:
    """Weighted experience score based on traveller preferences."""
    if not attractions:
        return 0.0
    tag_weights = {
        "culture":    profile.pref_culture,
        "history":    profile.pref_culture,
        "art":        profile.pref_culture,
        "nature":     profile.pref_nature,
        "park":       profile.pref_nature,
        "food":       profile.pref_food,
        "restaurant": profile.pref_food,
        "adventure":  profile.pref_adventure,
        "sport":      profile.pref_adventure,
        "relaxation": profile.pref_relaxation,
        "spa":        profile.pref_relaxation,
        "beach":      profile.pref_relaxation,
    }
    total = 0.0
    for n in attractions:
        tag_boost = sum(tag_weights.get(t.lower(), 0) for t in n.tags)
        score = n.rating * n.weather_comfort * (1 + tag_boost * 0.2)
        total += score
    return total / (len(attractions) * 5)  # normalise to [0, 1]


# ─── pymoo Problem definition ─────────────────────────────────────────────────

class TripOptimizationProblem(Problem):
    """
    Binary-selection problem: which attractions to include in the trip.
    """

    def __init__(
        self,
        candidates: list[TravelNode],
        profile: TravelerProfile,
        travel_times_minutes: list[float],  # time from hotel to each attraction
    ) -> None:
        self.candidates = candidates
        self.profile = profile
        self.travel_times = travel_times_minutes
        n = len(candidates)
        super().__init__(
            n_var=n,
            n_obj=5,
            n_ieq_constr=2,          # budget + daily limit constraints
            xl=np.zeros(n),
            xu=np.ones(n),
        )

    def _evaluate(self, X: np.ndarray, out: dict, *args: Any, **kwargs: Any) -> None:
        n_sol = X.shape[0]
        F = np.zeros((n_sol, 5))
        G = np.zeros((n_sol, 2))   # inequality constraints ≤ 0

        max_daily = self.profile.trip_duration_days * 8  # available hours/day

        for i, x in enumerate(X):
            selected = [self.candidates[j] for j, v in enumerate(x) if v > 0.5]

            # Objectives
            exp = experience_score(selected, self.profile)
            cost = sum(n.entry_cost_usd for n in selected) + sum(
                self.travel_times[j] / 60 * 5  # $5/travel-hour proxy
                for j, v in enumerate(x) if v > 0.5
            )
            time_h = sum(
                n.avg_visit_duration_hours + self.travel_times[j] / 60
                for j, v in enumerate(x)
                if v > 0.5
                for n in [self.candidates[j]]
            )
            crowd = np.mean([n.current_crowd_level for n in selected]) if selected else 0
            fatigue = fatigue_score(selected)

            F[i] = [-exp, cost, time_h, crowd, fatigue]

            # Constraint 1: total cost ≤ budget
            G[i, 0] = cost - self.profile.budget_usd

            # Constraint 2: total hours ≤ available trip hours
            G[i, 1] = time_h - max_daily

        out["F"] = F
        out["G"] = G


# ─── Public optimizer ─────────────────────────────────────────────────────────

class TripOptimizer:
    """
    High-level optimizer: wraps NSGA-II and returns an Itinerary.
    """

    def __init__(
        self,
        pop_size: int = 80,
        n_gen: int = 150,
        seed: int = 42,
    ) -> None:
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.seed = seed

    def optimize(
        self,
        candidates: list[TravelNode],
        profile: TravelerProfile,
        travel_times_minutes: list[float],
        weights: OptimizationWeights | None = None,
        start_date: datetime | None = None,
    ) -> OptimizationResult:
        weights = weights or OptimizationWeights()
        start_date = start_date or datetime.utcnow()

        t0 = time.perf_counter()

        problem = TripOptimizationProblem(candidates, profile, travel_times_minutes)
        algorithm = NSGA2(pop_size=self.pop_size)
        termination = get_termination("n_gen", self.n_gen)

        res = minimize(problem, algorithm, termination, seed=self.seed, verbose=False)

        elapsed = time.perf_counter() - t0

        # ── Select best solution from Pareto front ─────────────────────────
        best_x, pareto_info = self._select_from_pareto(res.X, res.F, weights)

        selected = [candidates[j] for j, v in enumerate(best_x) if v > 0.5]
        itinerary = self._build_itinerary(selected, profile, start_date, travel_times_minutes, candidates)

        return OptimizationResult(
            itinerary=itinerary,
            pareto_solutions=pareto_info,
            optimization_time_seconds=elapsed,
            algorithm="NSGA-II",
            convergence_score=float(np.mean(res.F[:, 0])) if res.F is not None else 0.0,
        )

    def _select_from_pareto(
        self,
        X: np.ndarray,
        F: np.ndarray,
        weights: OptimizationWeights,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        """Weighted Tchebycheff scalarisation to pick one solution."""
        if X is None or len(X) == 0:
            return np.array([]), []

        # Normalise objectives to [0, 1]
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = np.where(F_max - F_min > 0, F_max - F_min, 1.0)
        F_norm = (F - F_min) / F_range

        # Weight vector aligned with our 5 objectives
        # [experience(neg), cost, time, crowd, fatigue]
        w = np.array([
            weights.experience,
            weights.cost,
            weights.time,
            weights.crowd,
            weights.fatigue,
        ])

        # Tchebycheff: minimise max weighted deviation from utopia
        scores = np.max(w * F_norm, axis=1)
        best_idx = int(np.argmin(scores))

        pareto_info = [
            {
                "experience": float(-row[0]),
                "cost_usd":   float(row[1]),
                "time_hours": float(row[2]),
                "crowd":      float(row[3]),
                "fatigue":    float(row[4]),
            }
            for row in F
        ]
        return X[best_idx], pareto_info

    def _build_itinerary(
        self,
        selected: list[TravelNode],
        profile: TravelerProfile,
        start_date: datetime,
        travel_times: list[float],
        all_candidates: list[TravelNode],
    ) -> Itinerary:
        """Pack selected attractions into day-plans respecting daily hours."""
        days: list[DayPlan] = []
        stops_per_day: list[list[TravelNode]] = []

        # Greedy daily packing
        current_day: list[TravelNode] = []
        day_hours = 0.0
        daily_budget = profile.budget_usd / max(profile.trip_duration_days, 1)

        for node in selected:
            visit_h = node.avg_visit_duration_hours
            idx = all_candidates.index(node)
            travel_h = travel_times[idx] / 60
            if day_hours + visit_h + travel_h > 10 or len(current_day) >= 5:
                stops_per_day.append(current_day)
                current_day = []
                day_hours = 0.0
            current_day.append(node)
            day_hours += visit_h + travel_h

        if current_day:
            stops_per_day.append(current_day)

        total_cost = 0.0
        for day_idx, stops in enumerate(stops_per_day):
            date = start_date + timedelta(days=day_idx)
            day_start = datetime.combine(date.date(), profile.daily_start_time)
            day_cost = sum(n.entry_cost_usd for n in stops)
            total_cost += day_cost

            itinerary_stops = []
            cursor = day_start
            for node in stops:
                arrival = cursor
                departure = arrival + timedelta(hours=node.avg_visit_duration_hours)
                itinerary_stops.append(
                    ItineraryStop(
                        node=node,
                        arrival_time=arrival,
                        departure_time=departure,
                    )
                )
                cursor = departure + timedelta(minutes=20)  # transit buffer

            days.append(
                DayPlan(
                    day=day_idx + 1,
                    date=date,
                    stops=itinerary_stops,
                    estimated_cost_usd=day_cost,
                    fatigue_score=fatigue_score(stops),
                )
            )

        city = selected[0].city if selected else "Unknown"
        return Itinerary(
            profile=profile,
            destination_city=city,
            start_date=start_date,
            days=days,
            total_cost_usd=total_cost,
            overall_score=experience_score(selected, profile),
        )
