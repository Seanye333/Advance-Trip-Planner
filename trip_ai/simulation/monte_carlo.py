"""
Monte Carlo Risk Simulator
===========================
Runs N simulated trips to estimate:
  - Success rate (completing the plan without major disruptions)
  - Average extra time/cost from disruptions
  - Worst-case scenarios
  - High-risk segments (attractions or transit legs most likely to cause issues)

Disruption types and default probabilities:
  ┌──────────────────────────┬───────────┬────────────┬──────────────┐
  │ Event                    │ Prob/trip │ Time lost  │ Extra cost $ │
  ├──────────────────────────┼───────────┼────────────┼──────────────┤
  │ Flight delay (short)     │ 0.20      │ 1–3 h      │ 0–50         │
  │ Flight delay (long/miss) │ 0.05      │ 4–12 h     │ 100–500      │
  │ Attraction closure       │ 0.10      │ 0.5–2 h    │ 0–30         │
  │ Weather disruption       │ 0.15      │ 1–4 h      │ 20–100       │
  │ Traffic spike            │ 0.25      │ 0.3–1 h    │ 0            │
  │ Overbooking (hotel)      │ 0.03      │ 2–4 h      │ 50–200       │
  │ Illness / fatigue        │ 0.08      │ 2–8 h      │ 20–80        │
  │ Strike / political       │ 0.02      │ 4–24 h     │ 100–300      │
  └──────────────────────────┴───────────┴────────────┴──────────────┘
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean, quantiles
from uuid import UUID

from trip_ai.core.models import (
    Itinerary,
    RiskLevel,
    RiskScenario,
    SimulationResult,
    EventType,
)


# ─── Disruption catalogue ─────────────────────────────────────────────────────

DEFAULT_SCENARIOS: list[RiskScenario] = [
    RiskScenario(
        name="Short flight delay",
        probability=0.20,
        impact_hours=2.0,
        impact_cost_usd=25.0,
        event_type=EventType.FLIGHT_DELAY,
    ),
    RiskScenario(
        name="Missed connection",
        probability=0.05,
        impact_hours=8.0,
        impact_cost_usd=300.0,
        event_type=EventType.FLIGHT_DELAY,
    ),
    RiskScenario(
        name="Attraction closed / sold out",
        probability=0.10,
        impact_hours=1.0,
        impact_cost_usd=15.0,
        event_type=EventType.CLOSURE,
    ),
    RiskScenario(
        name="Severe weather",
        probability=0.15,
        impact_hours=2.5,
        impact_cost_usd=60.0,
        event_type=EventType.WEATHER_CHANGE,
    ),
    RiskScenario(
        name="Traffic / transit delay",
        probability=0.25,
        impact_hours=0.6,
        impact_cost_usd=0.0,
        event_type=EventType.TRAFFIC_SPIKE,
    ),
    RiskScenario(
        name="Hotel overbooking",
        probability=0.03,
        impact_hours=3.0,
        impact_cost_usd=120.0,
        event_type=EventType.FLIGHT_DELAY,  # closest proxy
    ),
    RiskScenario(
        name="Fatigue / illness",
        probability=0.08,
        impact_hours=5.0,
        impact_cost_usd=50.0,
        event_type=EventType.FATIGUE,
    ),
    RiskScenario(
        name="Strike / political event",
        probability=0.02,
        impact_hours=12.0,
        impact_cost_usd=200.0,
        event_type=EventType.STRIKE,
    ),
]


@dataclass
class TripSimRun:
    success: bool          # completed without major disruption (>4h lost)
    total_delay_hours: float
    extra_cost_usd: float
    triggered_events: list[str]


# ─── Simulator ────────────────────────────────────────────────────────────────

class MonteCarloSimulator:
    """
    Stateless Monte Carlo engine for risk quantification.
    """

    def __init__(
        self,
        scenarios: list[RiskScenario] | None = None,
        n_iterations: int = 10_000,
        seed: int | None = None,
    ) -> None:
        self.scenarios = scenarios or DEFAULT_SCENARIOS
        self.n_iterations = n_iterations
        self._rng = random.Random(seed)

    def simulate(self, itinerary: Itinerary) -> SimulationResult:
        """Run full Monte Carlo simulation for the given itinerary."""
        runs: list[TripSimRun] = [
            self._simulate_one(itinerary) for _ in range(self.n_iterations)
        ]

        successes = [r for r in runs if r.success]
        success_rate = len(successes) / self.n_iterations

        delays = [r.total_delay_hours for r in runs]
        costs  = [r.extra_cost_usd for r in runs]

        # Risk hot spots: events that triggered most often
        event_counts: dict[str, int] = {}
        for r in runs:
            for ev in r.triggered_events:
                event_counts[ev] = event_counts.get(ev, 0) + 1
        risk_hotspots = sorted(event_counts, key=lambda k: -event_counts[k])[:3]

        avg_delay = mean(delays)
        worst = max(delays)
        risk_level = self._classify_risk(success_rate, worst)

        return SimulationResult(
            itinerary_id=itinerary.id,
            n_iterations=self.n_iterations,
            success_rate=success_rate,
            avg_delay_hours=avg_delay,
            avg_extra_cost_usd=mean(costs),
            worst_case_delay_hours=worst,
            risk_hotspots=risk_hotspots,
            risk_level=risk_level,
        )

    def _simulate_one(self, itinerary: Itinerary) -> TripSimRun:
        """Simulate a single trip run."""
        total_delay = 0.0
        extra_cost = 0.0
        triggered: list[str] = []

        # Risk scales with trip duration and number of stops
        n_stops = sum(len(d.stops) for d in itinerary.days)
        duration_factor = min(itinerary.profile.trip_duration_days / 7, 1.5)
        complexity_factor = min(n_stops / 15, 1.5)

        for scenario in self.scenarios:
            adjusted_prob = scenario.probability * duration_factor * complexity_factor
            if self._rng.random() < adjusted_prob:
                # Randomise impact within ±50% of base value
                delay = scenario.impact_hours * self._rng.uniform(0.5, 1.5)
                cost  = scenario.impact_cost_usd * self._rng.uniform(0.5, 1.5)
                total_delay += delay
                extra_cost  += cost
                triggered.append(scenario.name)

        success = total_delay < 4.0  # >4 hours lost = "disrupted trip"
        return TripSimRun(
            success=success,
            total_delay_hours=total_delay,
            extra_cost_usd=extra_cost,
            triggered_events=triggered,
        )

    @staticmethod
    def _classify_risk(success_rate: float, worst_delay: float) -> RiskLevel:
        if success_rate >= 0.90 and worst_delay < 6:
            return RiskLevel.LOW
        if success_rate >= 0.75 and worst_delay < 12:
            return RiskLevel.MEDIUM
        if success_rate >= 0.55:
            return RiskLevel.HIGH
        return RiskLevel.EXTREME

    def sensitivity_analysis(
        self,
        itinerary: Itinerary,
        n_samples: int = 500,
    ) -> dict[str, float]:
        """
        Sobol-style one-at-a-time sensitivity: which scenario has the biggest
        impact on success rate when its probability is doubled?
        Returns {scenario_name: delta_success_rate}.
        """
        baseline = self.simulate(itinerary).success_rate
        results: dict[str, float] = {}

        for i, scenario in enumerate(self.scenarios):
            modified = list(self.scenarios)
            modified[i] = RiskScenario(
                name=scenario.name,
                probability=min(scenario.probability * 2, 1.0),
                impact_hours=scenario.impact_hours,
                impact_cost_usd=scenario.impact_cost_usd,
                event_type=scenario.event_type,
            )
            sim = MonteCarloSimulator(
                scenarios=modified,
                n_iterations=n_samples,
                seed=42,
            )
            delta = baseline - sim.simulate(itinerary).success_rate
            results[scenario.name] = round(delta, 4)

        return dict(sorted(results.items(), key=lambda x: -x[1]))
