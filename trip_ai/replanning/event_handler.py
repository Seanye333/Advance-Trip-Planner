"""
Disruption Event Handler
=========================
Detects and classifies real-time disruption events, then determines
the *severity* and *scope* of replanning required.

Disruption taxonomy:
  ┌──────────────────┬────────────────────────────────────────────────┐
  │ EventType        │ Replanning scope                               │
  ├──────────────────┼────────────────────────────────────────────────┤
  │ FLIGHT_DELAY     │ Current day + transit legs only                │
  │ WEATHER_CHANGE   │ Current day outdoor attractions → swap indoors │
  │ CLOSURE          │ Remove affected node, fill gap                 │
  │ TRAFFIC_SPIKE    │ Reschedule next 2–3 stops                      │
  │ FATIGUE          │ Trim current day, add rest stop                │
  │ BUDGET_EXCEEDED  │ Swap expensive stops for free alternatives     │
  │ STRIKE           │ Full replanning may be needed                  │
  └──────────────────┴────────────────────────────────────────────────┘
"""
from __future__ import annotations

from datetime import datetime

from trip_ai.core.models import DisruptionEvent, EventType, Itinerary


class DisruptionEventHandler:
    """Receives events, validates them, and enriches with context."""

    def create_flight_delay(
        self,
        delay_hours: float,
        flight_node_id: str,
        occurred_at: datetime | None = None,
    ) -> DisruptionEvent:
        severity = min(delay_hours / 12, 1.0)  # 12h delay = max severity
        return DisruptionEvent(
            event_type=EventType.FLIGHT_DELAY,
            affected_node_id=flight_node_id,
            occurred_at=occurred_at or datetime.utcnow(),
            severity=severity,
            details={"delay_hours": delay_hours},
        )

    def create_weather_event(
        self,
        severity: float,
        affected_city: str,
        weather_type: str = "rain",
    ) -> DisruptionEvent:
        return DisruptionEvent(
            event_type=EventType.WEATHER_CHANGE,
            occurred_at=datetime.utcnow(),
            severity=severity,
            details={"city": affected_city, "weather_type": weather_type},
        )

    def create_closure(
        self,
        node_id: str,
        reason: str = "unexpected",
    ) -> DisruptionEvent:
        return DisruptionEvent(
            event_type=EventType.CLOSURE,
            affected_node_id=node_id,
            occurred_at=datetime.utcnow(),
            severity=0.4,
            details={"reason": reason},
        )

    def create_fatigue_event(
        self,
        current_energy: float,
    ) -> DisruptionEvent:
        """current_energy: 0=exhausted, 1=full energy"""
        severity = 1.0 - current_energy
        return DisruptionEvent(
            event_type=EventType.FATIGUE,
            occurred_at=datetime.utcnow(),
            severity=severity,
            details={"energy_level": current_energy},
        )

    def create_budget_alert(
        self,
        spent_usd: float,
        budget_usd: float,
    ) -> DisruptionEvent:
        over_pct = (spent_usd - budget_usd) / budget_usd
        return DisruptionEvent(
            event_type=EventType.BUDGET_EXCEEDED,
            occurred_at=datetime.utcnow(),
            severity=min(over_pct, 1.0),
            details={"spent_usd": spent_usd, "budget_usd": budget_usd},
        )

    def requires_full_replan(self, event: DisruptionEvent) -> bool:
        """Does this event need a complete re-optimisation?"""
        return (
            event.event_type == EventType.STRIKE or
            (event.event_type == EventType.FLIGHT_DELAY and event.severity > 0.7) or
            (event.event_type == EventType.WEATHER_CHANGE and event.severity > 0.8)
        )

    def affected_day(
        self,
        event: DisruptionEvent,
        itinerary: Itinerary,
    ) -> int | None:
        """Return the day number (1-indexed) most affected, or None if unknown."""
        if not event.affected_node_id:
            return 1  # assume current day
        for day in itinerary.days:
            for stop in day.stops:
                if stop.node.id == event.affected_node_id:
                    return day.day
        return None
