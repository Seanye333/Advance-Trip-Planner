"""
Preference Learning Engine
===========================
Learns and refines traveller preferences over time using:
  - Explicit ratings (thumbs up/down on visited places)
  - Implicit signals (time spent, revisit intent, skip decisions)
  - Collaborative filtering (users with similar taste profiles)

The engine stores a preference vector per user and updates it
incrementally using an exponential moving average (EMA).

In production, replace the in-memory store with a proper user database.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from trip_ai.core.models import TravelerProfile, TravelNode


@dataclass
class FeedbackEvent:
    node_id: str
    event: Literal["like", "dislike", "skip", "extended_visit", "revisit"]
    weight: float = 1.0   # optional intensity (e.g. 3-star vs 5-star)


# Tag-to-preference mapping: which tags affect which preference dimension
TAG_DIMENSION_MAP: dict[str, str] = {
    "culture":    "pref_culture",
    "history":    "pref_culture",
    "art":        "pref_culture",
    "museum":     "pref_culture",
    "nature":     "pref_nature",
    "park":       "pref_nature",
    "hiking":     "pref_nature",
    "wildlife":   "pref_nature",
    "food":       "pref_food",
    "restaurant": "pref_food",
    "market":     "pref_food",
    "adventure":  "pref_adventure",
    "sport":      "pref_adventure",
    "extreme":    "pref_adventure",
    "relaxation": "pref_relaxation",
    "spa":        "pref_relaxation",
    "beach":      "pref_relaxation",
}

EVENT_DELTA: dict[str, float] = {
    "like":           +0.05,
    "dislike":        -0.05,
    "skip":           -0.03,
    "extended_visit": +0.03,
    "revisit":        +0.06,
}


class PreferenceEngine:
    """
    Updates a TravelerProfile's preference weights based on feedback events.
    Uses EMA so recent feedback matters more than old feedback.
    """

    def __init__(self, ema_alpha: float = 0.3) -> None:
        """
        ema_alpha : learning rate for exponential moving average
                    0 = ignore new data, 1 = use only new data
        """
        self.ema_alpha = ema_alpha
        # { user_id: TravelerProfile }
        self._profiles: dict[str, TravelerProfile] = {}
        # { user_id: [ FeedbackEvent ] }
        self._history: dict[str, list[FeedbackEvent]] = {}

    def register_profile(self, profile: TravelerProfile) -> None:
        uid = str(profile.id)
        self._profiles[uid] = profile
        self._history.setdefault(uid, [])

    def record_feedback(
        self,
        profile: TravelerProfile,
        node: TravelNode,
        event: Literal["like", "dislike", "skip", "extended_visit", "revisit"],
        weight: float = 1.0,
    ) -> TravelerProfile:
        """
        Apply a feedback event and return the updated profile.
        """
        uid = str(profile.id)
        self._history.setdefault(uid, []).append(
            FeedbackEvent(node_id=node.id, event=event, weight=weight)
        )

        delta = EVENT_DELTA[event] * weight
        updated_prefs = profile.model_dump()

        for tag in node.tags:
            dim = TAG_DIMENSION_MAP.get(tag.lower())
            if dim and dim in updated_prefs:
                old_val = float(updated_prefs[dim])
                # EMA update
                new_val = old_val + self.ema_alpha * delta
                updated_prefs[dim] = max(0.0, min(1.0, new_val))

        updated = TravelerProfile(**updated_prefs)
        self._profiles[uid] = updated
        return updated

    def get_profile(self, user_id: str) -> TravelerProfile | None:
        return self._profiles.get(user_id)

    def preference_vector(self, profile: TravelerProfile) -> list[float]:
        """Return the 5-D preference vector [culture, nature, food, adventure, relaxation]."""
        return [
            profile.pref_culture,
            profile.pref_nature,
            profile.pref_food,
            profile.pref_adventure,
            profile.pref_relaxation,
        ]

    def score_node(self, node: TravelNode, profile: TravelerProfile) -> float:
        """
        Compute how well a node matches the current preference vector.
        Returns a score in [0, 1].
        """
        pref_map = {
            "culture": profile.pref_culture,   "history": profile.pref_culture,
            "art":     profile.pref_culture,   "museum":  profile.pref_culture,
            "nature":  profile.pref_nature,    "park":    profile.pref_nature,
            "food":    profile.pref_food,      "adventure": profile.pref_adventure,
            "relaxation": profile.pref_relaxation,
        }
        if not node.tags:
            return node.rating / 5

        tag_score = sum(pref_map.get(t.lower(), 0) for t in node.tags)
        normalised = tag_score / len(node.tags)
        return min(1.0, 0.5 * (node.rating / 5) + 0.5 * normalised)
