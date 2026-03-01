"""
Real-Time Replanning Engine
============================
When a DisruptionEvent fires, this engine re-optimises the remaining
itinerary in seconds — like Google Maps rerouting after a road closure.

Strategy per event type:
  FLIGHT_DELAY     → shift all subsequent timings by delay_hours
  CLOSURE          → remove node, insert best alternative from candidates
  WEATHER_CHANGE   → swap outdoor attractions for indoor equivalents
  FATIGUE          → remove the last 1–2 stops of the current day
  BUDGET_EXCEEDED  → replace highest-cost remaining stops with free ones
  TRAFFIC_SPIKE    → add buffer time between current and next stop
  STRIKE           → trigger full re-optimisation pipeline
"""
from __future__ import annotations

import time
from copy import deepcopy
from datetime import datetime, timedelta

from trip_ai.core.models import (
    DisruptionEvent,
    EventType,
    Itinerary,
    ItineraryStop,
    ReplanResult,
)
from trip_ai.graph_engine.travel_graph import TravelGraph
from trip_ai.replanning.event_handler import DisruptionEventHandler


class ReplanEngine:
    """
    Fast, event-driven replanning.
    Each handler method modifies a deep copy of the itinerary and returns
    a ReplanResult with a summary of changes made.
    """

    def __init__(self, graph: TravelGraph) -> None:
        self.graph = graph
        self.event_handler = DisruptionEventHandler()

    def replan(
        self,
        itinerary: Itinerary,
        event: DisruptionEvent,
        current_day: int = 1,
    ) -> ReplanResult:
        """Dispatch to the right replanning strategy."""
        t0 = time.perf_counter()
        new_itinerary = deepcopy(itinerary)
        changes: list[str] = []

        dispatch = {
            EventType.FLIGHT_DELAY:    self._handle_flight_delay,
            EventType.CLOSURE:         self._handle_closure,
            EventType.WEATHER_CHANGE:  self._handle_weather,
            EventType.FATIGUE:         self._handle_fatigue,
            EventType.BUDGET_EXCEEDED: self._handle_budget,
            EventType.TRAFFIC_SPIKE:   self._handle_traffic,
            EventType.STRIKE:          self._handle_strike,
        }

        handler = dispatch.get(event.event_type)
        if handler:
            new_itinerary, changes = handler(new_itinerary, event, current_day)
        else:
            changes = [f"Unhandled event type: {event.event_type}"]

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return ReplanResult(
            original_itinerary_id=itinerary.id,
            new_itinerary=new_itinerary,
            trigger_event=event,
            changes_summary=changes,
            replan_time_ms=elapsed_ms,
        )

    # ─── Individual handlers ──────────────────────────────────────────────────

    def _handle_flight_delay(
        self,
        itinerary: Itinerary,
        event: DisruptionEvent,
        current_day: int,
    ) -> tuple[Itinerary, list[str]]:
        delay_h = float(event.details.get("delay_hours", 2.0))
        changes = [f"Flight delayed by {delay_h:.1f}h — shifting all future stops"]

        # Shift every stop from current_day onwards
        for day in itinerary.days:
            if day.day >= current_day:
                for stop in day.stops:
                    stop.arrival_time   += timedelta(hours=delay_h)
                    stop.departure_time += timedelta(hours=delay_h)

        # If delay > 4h, drop the last stop of current day to avoid overrun
        if delay_h >= 4 and current_day <= len(itinerary.days):
            day = itinerary.days[current_day - 1]
            if day.stops:
                dropped = day.stops.pop()
                changes.append(f"Dropped '{dropped.node.name}' to absorb delay")

        return itinerary, changes

    def _handle_closure(
        self,
        itinerary: Itinerary,
        event: DisruptionEvent,
        current_day: int,
    ) -> tuple[Itinerary, list[str]]:
        node_id = event.affected_node_id
        changes = []

        for day in itinerary.days:
            original_len = len(day.stops)
            day.stops = [s for s in day.stops if s.node.id != node_id]
            if len(day.stops) < original_len:
                changes.append(f"Removed closed attraction (id={node_id}) from Day {day.day}")

                # Try to insert an alternative from the same city
                candidates = self.graph.nodes_in_city(itinerary.destination_city)
                visited_ids = {s.node.id for d in itinerary.days for s in d.stops}
                alternatives = [
                    n for n in candidates
                    if n.id not in visited_ids and n.id != node_id
                    and n.entry_cost_usd <= itinerary.profile.budget_usd * 0.1
                ]
                if alternatives:
                    best = max(alternatives, key=lambda n: n.rating)
                    last_stop = day.stops[-1] if day.stops else None
                    arrival = (last_stop.departure_time + timedelta(minutes=20)) if last_stop else \
                              datetime.combine(day.date.date(), itinerary.profile.daily_start_time)
                    departure = arrival + timedelta(hours=best.avg_visit_duration_hours)
                    day.stops.append(
                        ItineraryStop(
                            node=best,
                            arrival_time=arrival,
                            departure_time=departure,
                        )
                    )
                    changes.append(f"Inserted alternative: '{best.name}'")

        return itinerary, changes

    def _handle_weather(
        self,
        itinerary: Itinerary,
        event: DisruptionEvent,
        current_day: int,
    ) -> tuple[Itinerary, list[str]]:
        changes = []
        weather = event.details.get("weather_type", "rain")

        # Swap outdoor attractions for indoor ones
        indoor_tags = {"museum", "art", "culture", "history", "restaurant", "food"}
        city_nodes = self.graph.nodes_in_city(itinerary.destination_city)
        indoor_pool = [
            n for n in city_nodes
            if set(t.lower() for t in n.tags) & indoor_tags
        ]

        for day in itinerary.days:
            if day.day < current_day:
                continue
            for i, stop in enumerate(day.stops):
                outdoor_tags = {"nature", "park", "beach", "adventure", "hiking"}
                if set(t.lower() for t in stop.node.tags) & outdoor_tags:
                    visited = {s.node.id for d in itinerary.days for s in d.stops}
                    candidates = [n for n in indoor_pool if n.id not in visited]
                    if candidates:
                        replacement = max(candidates, key=lambda n: n.rating)
                        duration = stop.departure_time - stop.arrival_time
                        new_stop = ItineraryStop(
                            node=replacement,
                            arrival_time=stop.arrival_time,
                            departure_time=stop.arrival_time + duration,
                        )
                        day.stops[i] = new_stop
                        changes.append(
                            f"Swapped outdoor '{stop.node.name}' → indoor '{replacement.name}' ({weather})"
                        )

        return itinerary, changes

    def _handle_fatigue(
        self,
        itinerary: Itinerary,
        event: DisruptionEvent,
        current_day: int,
    ) -> tuple[Itinerary, list[str]]:
        energy = float(event.details.get("energy_level", 0.3))
        changes = []

        if current_day <= len(itinerary.days):
            day = itinerary.days[current_day - 1]
            drops = max(1, int((1 - energy) * len(day.stops) * 0.4))
            for _ in range(drops):
                if day.stops:
                    dropped = day.stops.pop()
                    changes.append(f"Removed '{dropped.node.name}' to allow rest (energy {energy:.0%})")

        return itinerary, changes

    def _handle_budget(
        self,
        itinerary: Itinerary,
        event: DisruptionEvent,
        current_day: int,
    ) -> tuple[Itinerary, list[str]]:
        overspend = float(event.details.get("spent_usd", 0))
        changes = []

        # Remove the most expensive remaining stop
        for day in itinerary.days:
            if day.day <= current_day:
                continue
            if not day.stops:
                continue
            priciest = max(day.stops, key=lambda s: s.node.entry_cost_usd)
            if priciest.node.entry_cost_usd > 0:
                day.stops.remove(priciest)
                changes.append(
                    f"Removed expensive stop '{priciest.node.name}' "
                    f"(${priciest.node.entry_cost_usd:.0f}) to stay in budget"
                )
                break

        return itinerary, changes

    def _handle_traffic(
        self,
        itinerary: Itinerary,
        event: DisruptionEvent,
        current_day: int,
    ) -> tuple[Itinerary, list[str]]:
        buffer_min = int(event.severity * 45)  # up to 45-min buffer
        changes = [f"Adding {buffer_min}-min traffic buffer to current-day stops"]

        if current_day <= len(itinerary.days):
            day = itinerary.days[current_day - 1]
            for stop in day.stops:
                stop.arrival_time   += timedelta(minutes=buffer_min)
                stop.departure_time += timedelta(minutes=buffer_min)

        return itinerary, changes

    def _handle_strike(
        self,
        itinerary: Itinerary,
        event: DisruptionEvent,
        current_day: int,
    ) -> tuple[Itinerary, list[str]]:
        changes = [
            "Strike detected — flagging for full re-optimisation.",
            "All transit-dependent legs marked as uncertain.",
            "Recommend switching to walking/taxi alternatives.",
        ]
        # Mark high-transit days as uncertain (simplified)
        for day in itinerary.days:
            if day.day >= current_day:
                for stop in day.stops:
                    stop.notes = "⚠ Transit uncertain due to strike. Verify before travelling."
        return itinerary, changes
