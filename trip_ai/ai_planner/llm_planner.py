"""
LLM Planner — Claude API Integration
======================================
Responsibilities:
  1. Generate a reasoning-based draft itinerary from user preferences
  2. Explain the final optimised itinerary in natural language
  3. Answer follow-up traveller questions (chat mode)
  4. Generate per-day narrative guides

Uses Claude claude-sonnet-4-6 via the Anthropic SDK.
The LLM does *reasoning*; the math is handled by the optimizer modules.
"""
from __future__ import annotations

import json
from typing import Iterator

import anthropic

from trip_ai.core.config import settings
from trip_ai.core.models import Itinerary, TravelerProfile, TravelNode


SYSTEM_PROMPT = """You are TripAI — an expert AI travel planner powered by a real-time
optimization engine. Your role is to:
1. Reason about travel preferences and generate thoughtful itinerary drafts
2. Explain optimised itineraries in engaging, human-readable language
3. Flag potential issues (crowds, weather, budget conflicts) proactively
4. Answer travel questions with accuracy and enthusiasm

Important rules:
- Always be specific: mention real-sounding place names, timings, and practical tips
- When explaining an optimised plan, reference the objective trade-offs made
- Keep responses concise but rich — bullet points for day plans, paragraphs for narratives
- Never invent prices or opening hours without flagging uncertainty
"""


class LLMPlanner:
    """Thin wrapper around the Anthropic SDK for trip-planning tasks."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.claude_model
        self._conversation_history: list[dict] = []

    # ─── Core planning ────────────────────────────────────────────────────────

    def draft_itinerary(
        self,
        profile: TravelerProfile,
        city: str,
        candidate_attractions: list[TravelNode],
    ) -> str:
        """
        Ask Claude to reason about the best set of attractions and propose
        a draft visit order before math optimisation runs.
        """
        attraction_list = "\n".join(
            f"- {n.name} (rating {n.rating:.1f}, ${n.entry_cost_usd:.0f}, "
            f"~{n.avg_visit_duration_hours:.1f}h, tags: {', '.join(n.tags)})"
            for n in candidate_attractions[:20]  # cap context size
        )

        prompt = f"""I'm planning a {profile.trip_duration_days}-day trip to {city}.

TRAVELLER PROFILE:
- Budget: ${profile.budget_usd:.0f} total
- Interests: culture={profile.pref_culture:.0%}, nature={profile.pref_nature:.0%},
  food={profile.pref_food:.0%}, adventure={profile.pref_adventure:.0%}
- Mobility restricted: {profile.mobility_restricted}
- Dietary restrictions: {profile.dietary_restrictions or 'none'}
- Risk tolerance: {profile.risk_tolerance:.0%}

AVAILABLE ATTRACTIONS:
{attraction_list}

Please suggest which attractions to prioritise and in what rough order,
considering the traveller's preferences and practical logistics (proximity, timing).
Return your reasoning as bullet points, then a suggested day-by-day outline."""

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def explain_itinerary(self, itinerary: Itinerary) -> str:
        """
        Generate a rich narrative explanation of the optimised itinerary,
        including why trade-offs were made.
        """
        days_summary = []
        for day in itinerary.days:
            stops = " → ".join(s.node.name for s in day.stops)
            days_summary.append(
                f"Day {day.day}: {stops} | "
                f"Cost: ${day.estimated_cost_usd:.0f} | "
                f"Fatigue: {day.fatigue_score:.0%}"
            )

        prompt = f"""Here is an AI-optimised {itinerary.profile.trip_duration_days}-day
trip to {itinerary.destination_city} (overall score: {itinerary.overall_score:.0%}):

{chr(10).join(days_summary)}

Total cost: ${itinerary.total_cost_usd:.0f} (budget: ${itinerary.profile.budget_usd:.0f})
Risk level: {itinerary.risk_level.value}

Please write an engaging 3-4 paragraph explanation of this itinerary:
1. Why these attractions were selected
2. How the day flow was optimised (crowds, energy, geography)
3. What trade-offs the optimizer made (cost vs experience vs time)
4. Practical tips for each day"""

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def day_guide(self, itinerary: Itinerary, day_number: int) -> str:
        """Generate a detailed narrative guide for a single day."""
        if day_number < 1 or day_number > len(itinerary.days):
            return "Invalid day number."

        day = itinerary.days[day_number - 1]
        stops_detail = "\n".join(
            f"- {s.node.name}: arrive {s.arrival_time.strftime('%H:%M')}, "
            f"leave {s.departure_time.strftime('%H:%M')}, "
            f"${s.node.entry_cost_usd:.0f}"
            for s in day.stops
        )

        prompt = f"""Write a detailed Day {day_number} travel guide for {itinerary.destination_city}.

SCHEDULE:
{stops_detail}

Include: what to see at each stop, insider tips, where to eat between visits,
what to watch out for (queues, weather, scams), and how to get between stops.
Keep it practical and specific. Format with stop headers."""

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    # ─── Chat mode ────────────────────────────────────────────────────────────

    def chat(self, user_message: str, itinerary: Itinerary | None = None) -> str:
        """
        Stateful conversation with memory of past exchanges.
        Optionally inject current itinerary as context.
        """
        if itinerary and not self._conversation_history:
            # Inject itinerary context at the start of conversation
            context = (
                f"[Current itinerary: {itinerary.profile.trip_duration_days} days in "
                f"{itinerary.destination_city}, ${itinerary.total_cost_usd:.0f} total, "
                f"risk={itinerary.risk_level.value}]"
            )
            self._conversation_history.append(
                {"role": "user", "content": context}
            )
            self._conversation_history.append(
                {"role": "assistant", "content": "Got it! I have your itinerary loaded. How can I help?"}
            )

        self._conversation_history.append({"role": "user", "content": user_message})

        response = self._client.messages.create(
            model=self._model,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=self._conversation_history,
        )
        reply = response.content[0].text
        self._conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def chat_stream(self, user_message: str) -> Iterator[str]:
        """Streaming version of chat — yields text chunks as they arrive."""
        self._conversation_history.append({"role": "user", "content": user_message})

        with self._client.messages.stream(
            model=self._model,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=self._conversation_history,
        ) as stream:
            full_reply = ""
            for text in stream.text_stream:
                full_reply += text
                yield text

        self._conversation_history.append({"role": "assistant", "content": full_reply})

    def reset_chat(self) -> None:
        self._conversation_history.clear()

    # ─── Structured output ────────────────────────────────────────────────────

    def extract_preferences_from_text(self, text: str) -> dict:
        """
        Parse a free-form traveller description into structured preferences.
        Uses Claude with JSON output mode.
        """
        prompt = f"""Extract travel preferences from this text and return ONLY valid JSON.

Text: "{text}"

JSON schema:
{{
  "budget_usd": <number>,
  "trip_duration_days": <number>,
  "pref_culture": <0.0-1.0>,
  "pref_nature": <0.0-1.0>,
  "pref_food": <0.0-1.0>,
  "pref_adventure": <0.0-1.0>,
  "pref_relaxation": <0.0-1.0>,
  "risk_tolerance": <0.0-1.0>,
  "dietary_restrictions": [<string>, ...],
  "mobility_restricted": <bool>
}}
Fill in sensible defaults for any missing fields. Return ONLY the JSON object."""

        response = self._client.messages.create(
            model=self._model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return {}
