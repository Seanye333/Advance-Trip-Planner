"""
Crowd & Congestion Predictor
=============================
Predicts crowd levels for attractions and transit at a given time using:
  - Time-of-day curves (morning lull → midday peak → evening)
  - Day-of-week patterns (weekends ~30% busier)
  - Seasonal factors (school holidays, summer peak)
  - Event calendar (concerts, festivals, public holidays)
  - Historical base rates per attraction type

In production: train an XGBoost or LSTM model on real visitor-count data
(Google Popular Times, ticketing APIs, city open data).

Here we implement a deterministic rule-based model with added Gaussian noise
that can be replaced by a trained sklearn/XGBoost model via the same API.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from trip_ai.core.models import TravelNode, NodeType


# ─── Base crowd rates by node type ────────────────────────────────────────────

BASE_CROWD: dict[NodeType, float] = {
    NodeType.ATTRACTION: 0.55,
    NodeType.RESTAURANT: 0.50,
    NodeType.AIRPORT:    0.65,
    NodeType.TRANSPORT:  0.60,
    NodeType.HOTEL:      0.30,
    NodeType.CITY:       0.40,
}

# Seasonal multipliers (month → factor)
SEASONAL_FACTOR: dict[int, float] = {
    1: 0.70, 2: 0.72, 3: 0.80, 4: 0.90,
    5: 0.95, 6: 1.10, 7: 1.25, 8: 1.30,
    9: 1.05, 10: 0.90, 11: 0.75, 12: 0.85,
}


def _time_of_day_factor(hour: int) -> float:
    """
    Bell-curve-like crowd factor centred on midday (hour 12).
    Returns value in [0.3, 1.2].
    """
    # Two peaks: 11:00 and 15:00
    peak1 = math.exp(-0.5 * ((hour - 11) / 2.5) ** 2)
    peak2 = math.exp(-0.5 * ((hour - 15) / 2.0) ** 2)
    return 0.3 + 0.9 * max(peak1, peak2)


def _weekday_factor(weekday: int) -> float:
    """weekday: 0=Mon, 6=Sun"""
    # Weekend peaks
    factors = [0.85, 0.80, 0.85, 0.90, 1.00, 1.25, 1.30]
    return factors[weekday]


class CrowdPredictor:
    """
    Predict crowd level ∈ [0, 1] for a node at a given datetime.
    """

    def __init__(self, noise_std: float = 0.05) -> None:
        """
        noise_std : standard deviation of Gaussian noise added to predictions
                    (simulates real-world variability; set 0 for deterministic)
        """
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed=42)

    def predict(self, node: TravelNode, at: datetime) -> float:
        """Return predicted crowd level for *node* at datetime *at*."""
        base = BASE_CROWD.get(node.node_type, 0.5)
        tod  = _time_of_day_factor(at.hour)
        dow  = _weekday_factor(at.weekday())
        season = SEASONAL_FACTOR.get(at.month, 1.0)

        # Rating inversely correlates with crowd (popular = crowded)
        popularity_boost = (node.rating - 3.0) * 0.08  # -0.16 to +0.16

        crowd = base * tod * dow * season + popularity_boost
        noise = float(self._rng.normal(0, self.noise_std))
        return float(np.clip(crowd + noise, 0.0, 1.0))

    def best_visit_time(
        self,
        node: TravelNode,
        date: datetime,
        open_hour: int = 9,
        close_hour: int = 18,
    ) -> int:
        """Return the hour (24h) with the lowest predicted crowd on *date*."""
        hours = range(open_hour, close_hour)
        crowds = {
            h: self.predict(node, date.replace(hour=h, minute=0))
            for h in hours
        }
        return min(crowds, key=crowds.get)  # type: ignore

    def batch_predict(
        self,
        nodes: list[TravelNode],
        at: datetime,
    ) -> dict[str, float]:
        """Return {node_id: crowd_level} for a list of nodes."""
        return {n.id: self.predict(n, at) for n in nodes}

    def update_graph_crowds(
        self,
        nodes: list[TravelNode],
        at: datetime,
    ) -> list[TravelNode]:
        """
        Return copies of nodes with current_crowd_level updated.
        Call this before running the optimizer to get live-aware results.
        """
        updated = []
        for node in nodes:
            new_crowd = self.predict(node, at)
            data = node.model_dump()
            data["current_crowd_level"] = new_crowd
            updated.append(TravelNode(**data))
        return updated
