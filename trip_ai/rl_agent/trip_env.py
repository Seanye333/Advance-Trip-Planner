"""
Trip Planning Gymnasium Environment
=====================================
Models trip planning as a Markov Decision Process (MDP).

State  (continuous, 8-D):
  [budget_remaining_norm, time_remaining_norm, energy_norm,
   crowd_level, weather_score, visited_ratio,
   lat_norm, lon_norm]

Actions (discrete):
  0 = move to attraction 0
  1 = move to attraction 1
  ...
  N-1 = move to attraction N-1
  N   = take a rest (recover 0.15 energy)
  N+1 = end day early

Reward shaping:
  +rating × preference_match × weather_comfort   (visiting a good attraction)
  -entry_cost / budget_limit                      (spending penalty)
  -crowd_level × crowd_sensitivity                (crowd penalty)
  -travel_time_hours × 0.1                        (time cost)
  -0.2 × fatigue                                  (fatigue penalty)
  +2.0 on day completion with stops ≥ 3           (day bonus)
  -5.0 on budget bust                             (hard penalty)

Termination:
  - All days complete
  - Budget exhausted
  - Energy = 0 (collapse)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from trip_ai.core.models import NodeType, TravelerProfile, TravelNode
from trip_ai.graph_engine.travel_graph import TravelGraph, haversine_km


class TripPlanningEnv(gym.Env):
    """
    Custom Gymnasium environment for RL-based trip planning.

    Usage:
        env = TripPlanningEnv(attractions, profile, graph)
        obs, _ = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        attractions: list[TravelNode],
        profile: TravelerProfile,
        graph: TravelGraph,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.attractions = attractions
        self.profile = profile
        self.graph = graph
        self.render_mode = render_mode

        n = len(attractions)
        # Action: visit attraction i, take rest (n), end day (n+1)
        self.action_space = spaces.Discrete(n + 2)
        # Observation: 8-D continuous state
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self._tag_weights = {
            "culture": profile.pref_culture, "history": profile.pref_culture,
            "art": profile.pref_culture,    "museum":  profile.pref_culture,
            "nature": profile.pref_nature,  "park":    profile.pref_nature,
            "food":   profile.pref_food,    "adventure": profile.pref_adventure,
            "relaxation": profile.pref_relaxation,
        }

        # Lat/lon normalisation bounds
        if attractions:
            self._lat_min = min(n.lat for n in attractions)
            self._lat_max = max(n.lat for n in attractions)
            self._lon_min = min(n.lon for n in attractions)
            self._lon_max = max(n.lon for n in attractions)
        else:
            self._lat_min = self._lat_max = self._lon_min = self._lon_max = 0.0

        self._reset_state()

    # ─── Gymnasium interface ──────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        n = len(self.attractions)
        reward = 0.0
        info: dict[str, Any] = {}

        if action == n:  # REST
            self._energy = min(1.0, self._energy + 0.15)
            reward = -0.05  # small cost for resting (opportunity cost)

        elif action == n + 1:  # END DAY EARLY
            stops_today = self._stops_this_day
            if stops_today >= 3:
                reward += 2.0  # completed a good day
            self._start_new_day()

        elif self.attractions[action].id in self._visited:
            reward = -1.0  # penalty for revisiting

        else:
            node = self.attractions[action]
            if not self._can_afford(node) or not self._has_time(node):
                reward = -0.5
            else:
                reward = self._visit_attraction(node)

        # Check termination
        done = (
            self._current_day > self.profile.trip_duration_days or
            self._budget_remaining <= 0 or
            self._energy <= 0
        )

        obs = self._get_obs()
        return obs, float(reward), done, False, info

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            return (
                f"Day {self._current_day}/{self.profile.trip_duration_days} | "
                f"Budget: ${self._budget_remaining:.0f} | "
                f"Energy: {self._energy:.0%} | "
                f"Visited: {len(self._visited)}/{len(self.attractions)}"
            )
        return None

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        self._budget_remaining = self.profile.budget_usd
        self._time_remaining_hours = float(
            self.profile.trip_duration_days * 10  # 10h/day
        )
        self._energy = 1.0
        self._current_day = 1
        self._current_time = datetime.utcnow().replace(
            hour=self.profile.daily_start_time.hour, minute=0
        )
        self._current_lat = (
            sum(n.lat for n in self.attractions) / len(self.attractions)
            if self.attractions else 0.0
        )
        self._current_lon = (
            sum(n.lon for n in self.attractions) / len(self.attractions)
            if self.attractions else 0.0
        )
        self._visited: set[str] = set()
        self._stops_this_day = 0

    def _get_obs(self) -> np.ndarray:
        lat_range = max(self._lat_max - self._lat_min, 1e-6)
        lon_range = max(self._lon_max - self._lon_min, 1e-6)
        return np.array([
            self._budget_remaining / self.profile.budget_usd,
            self._time_remaining_hours / (self.profile.trip_duration_days * 10),
            self._energy,
            0.5,  # placeholder crowd (could be from CrowdPredictor)
            0.8,  # placeholder weather comfort
            len(self._visited) / max(len(self.attractions), 1),
            (self._current_lat - self._lat_min) / lat_range,
            (self._current_lon - self._lon_min) / lon_range,
        ], dtype=np.float32)

    def _preference_match(self, node: TravelNode) -> float:
        if not node.tags:
            return 0.5
        score = sum(self._tag_weights.get(t.lower(), 0) for t in node.tags)
        return min(1.0, score / len(node.tags))

    def _visit_attraction(self, node: TravelNode) -> float:
        travel_km = haversine_km(
            self._current_lat, self._current_lon, node.lat, node.lon
        )
        travel_h = travel_km / 5.0  # 5 km/h walking
        visit_h  = node.avg_visit_duration_hours
        total_h  = travel_h + visit_h

        # Update state
        self._budget_remaining  -= node.entry_cost_usd
        self._time_remaining_hours -= total_h
        self._energy = max(0, self._energy - (total_h * 0.08 + node.current_crowd_level * 0.05))
        self._current_lat = node.lat
        self._current_lon = node.lon
        self._visited.add(node.id)
        self._stops_this_day += 1

        # Compute reward
        preference = self._preference_match(node)
        reward = (
            node.rating * preference * node.weather_comfort
            - node.entry_cost_usd / self.profile.budget_usd
            - node.current_crowd_level * (1 - self.profile.crowd_tolerance)
            - travel_h * 0.1
            - (1 - self._energy) * 0.2
        )
        return reward

    def _start_new_day(self) -> None:
        self._current_day += 1
        self._stops_this_day = 0
        self._energy = min(1.0, self._energy + 0.3)  # overnight recovery

    def _can_afford(self, node: TravelNode) -> bool:
        return self._budget_remaining >= node.entry_cost_usd

    def _has_time(self, node: TravelNode) -> bool:
        return self._time_remaining_hours >= node.avg_visit_duration_hours + 0.5
