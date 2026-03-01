"""
MILP Itinerary Solver (PuLP)
=============================
Formulates daily itinerary scheduling as a Mixed-Integer Linear Program.

Decision variables:
  y[i]       = 1 if attraction i is visited, 0 otherwise
  order[i,j] = 1 if attraction i is visited immediately before j

Constraints:
  - Each selected attraction is visited exactly once
  - Total cost ≤ budget
  - Total time ≤ available hours
  - Opening-hours windows respected
  - Sub-tour elimination (Miller-Tucker-Zemlin)

Objective: maximise experience score (weighted sum of ratings × preferences)

Note: for >15 attractions use the NSGA-II optimizer instead — MILP scales
exponentially with the number of binary variables.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pulp

from trip_ai.core.models import (
    DayPlan,
    Itinerary,
    ItineraryStop,
    TravelerProfile,
    TravelNode,
)
from trip_ai.optimization.multi_objective import experience_score, fatigue_score


class ItinerarySolver:
    """
    MILP-based daily planner for small attraction sets (≤15 nodes).
    Returns a single optimal day plan.
    """

    def __init__(self, time_limit_seconds: int = 30) -> None:
        self.time_limit = time_limit_seconds

    def solve_day(
        self,
        candidates: list[TravelNode],
        profile: TravelerProfile,
        travel_matrix: list[list[float]],  # travel_matrix[i][j] = minutes from i to j
        date: datetime,
        day_number: int = 1,
        available_hours: float = 10.0,
        day_budget: float | None = None,
    ) -> DayPlan:
        """
        Solve a single-day scheduling problem.

        Parameters
        ----------
        candidates      : pool of attractions to choose from
        travel_matrix   : pairwise travel time in minutes (NxN)
        available_hours : hours available from start to end of day
        day_budget      : spending limit for the day
        """
        n = len(candidates)
        if n == 0:
            return DayPlan(day=day_number, date=date)

        budget = day_budget or (profile.budget_usd / profile.trip_duration_days)

        prob = pulp.LpProblem("day_planner", pulp.LpMaximize)

        # ── Decision variables ─────────────────────────────────────────────
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]
        # order[i][j] = 1 if we go directly from i to j
        order = [
            [pulp.LpVariable(f"o_{i}_{j}", cat="Binary") for j in range(n)]
            for i in range(n)
        ]
        # MTZ position variables for sub-tour elimination
        u = [pulp.LpVariable(f"u_{i}", lowBound=0, upBound=n) for i in range(n)]

        # ── Objective: maximise experience ─────────────────────────────────
        tag_weights = {
            "culture": profile.pref_culture, "history": profile.pref_culture,
            "nature": profile.pref_nature, "park": profile.pref_nature,
            "food": profile.pref_food, "adventure": profile.pref_adventure,
            "relaxation": profile.pref_relaxation,
        }

        def node_value(node: TravelNode) -> float:
            tag_boost = sum(tag_weights.get(t.lower(), 0) for t in node.tags)
            return node.rating * node.weather_comfort * (1 + tag_boost * 0.2)

        prob += pulp.lpSum(node_value(candidates[i]) * y[i] for i in range(n))

        # ── Constraints ────────────────────────────────────────────────────

        # Budget
        prob += (
            pulp.lpSum(candidates[i].entry_cost_usd * y[i] for i in range(n)) <= budget,
            "budget",
        )

        # Time: visit duration + travel time
        travel_hours = pulp.lpSum(
            (
                candidates[i].avg_visit_duration_hours * y[i] +
                pulp.lpSum(travel_matrix[i][j] / 60 * order[i][j] for j in range(n) if i != j)
            )
            for i in range(n)
        )
        prob += (travel_hours <= available_hours, "time_window")

        # Each visited node has exactly one successor (and predecessor)
        for i in range(n):
            prob += (
                pulp.lpSum(order[i][j] for j in range(n) if i != j) == y[i],
                f"out_{i}",
            )
            prob += (
                pulp.lpSum(order[j][i] for j in range(n) if i != j) == y[i],
                f"in_{i}",
            )
            # No self-loops
            prob += (order[i][i] == 0, f"no_self_{i}")

        # MTZ sub-tour elimination
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    prob += (
                        u[i] - u[j] + n * order[i][j] <= n - 1,
                        f"mtz_{i}_{j}",
                    )

        # Solve
        solver = pulp.PULP_CBC_CMD(
            timeLimit=self.time_limit,
            msg=False,
        )
        prob.solve(solver)

        # ── Extract solution ───────────────────────────────────────────────
        selected = [candidates[i] for i in range(n) if pulp.value(y[i]) and pulp.value(y[i]) > 0.5]

        # Order by MTZ position
        selected.sort(
            key=lambda node: pulp.value(u[candidates.index(node)]) or 0
        )

        stops: list[ItineraryStop] = []
        day_start = datetime.combine(date.date(), profile.daily_start_time)
        cursor = day_start
        day_cost = 0.0

        for node in selected:
            arrival = cursor
            departure = arrival + timedelta(hours=node.avg_visit_duration_hours)
            stops.append(
                ItineraryStop(node=node, arrival_time=arrival, departure_time=departure)
            )
            cursor = departure + timedelta(minutes=20)
            day_cost += node.entry_cost_usd

        return DayPlan(
            day=day_number,
            date=date,
            stops=stops,
            estimated_cost_usd=day_cost,
            fatigue_score=fatigue_score(selected),
        )
