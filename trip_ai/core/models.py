"""
Core Pydantic data models for the Trip AI platform.
Every module imports from here — single source of truth.
"""
from __future__ import annotations

from datetime import datetime, time
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    AIRPORT     = "airport"
    HOTEL       = "hotel"
    ATTRACTION  = "attraction"
    RESTAURANT  = "restaurant"
    TRANSPORT   = "transport_hub"
    CITY        = "city"


class TransportMode(str, Enum):
    FLIGHT  = "flight"
    TRAIN   = "train"
    BUS     = "bus"
    CAR     = "car"
    WALK    = "walk"
    SUBWAY  = "subway"
    FERRY   = "ferry"


class EventType(str, Enum):
    FLIGHT_DELAY    = "flight_delay"
    WEATHER_CHANGE  = "weather_change"
    CLOSURE         = "attraction_closure"
    TRAFFIC_SPIKE   = "traffic_spike"
    FATIGUE         = "traveler_fatigue"
    BUDGET_EXCEEDED = "budget_exceeded"
    STRIKE          = "strike"


class RiskLevel(str, Enum):
    LOW     = "low"
    MEDIUM  = "medium"
    HIGH    = "high"
    EXTREME = "extreme"


# ─── Graph Nodes ──────────────────────────────────────────────────────────────

class TravelNode(BaseModel):
    id: str
    name: str
    node_type: NodeType
    city: str
    country: str
    lat: float
    lon: float
    # Experience & logistics
    rating: float = Field(default=4.0, ge=0, le=5)
    avg_visit_duration_hours: float = Field(default=1.0, ge=0)
    entry_cost_usd: float = Field(default=0.0, ge=0)
    # Availability
    open_time: time = Field(default=time(9, 0))
    close_time: time = Field(default=time(18, 0))
    # Dynamic attributes (updated at runtime)
    current_crowd_level: float = Field(default=0.5, ge=0, le=1)
    weather_comfort: float = Field(default=0.8, ge=0, le=1)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TravelEdge(BaseModel):
    source_id: str
    target_id: str
    transport_mode: TransportMode
    duration_minutes: float
    cost_usd: float
    reliability: float = Field(default=0.95, ge=0, le=1)
    co2_kg: float = Field(default=0.0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── User Profile ─────────────────────────────────────────────────────────────

class TravelerProfile(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = "Traveler"
    budget_usd: float = Field(default=2000.0, ge=0)
    trip_duration_days: int = Field(default=5, ge=1)
    daily_start_time: time = Field(default=time(8, 0))
    daily_end_time: time = Field(default=time(22, 0))
    # Preferences (0–1 weights)
    pref_culture: float    = Field(default=0.7, ge=0, le=1)
    pref_nature: float     = Field(default=0.5, ge=0, le=1)
    pref_food: float       = Field(default=0.6, ge=0, le=1)
    pref_adventure: float  = Field(default=0.4, ge=0, le=1)
    pref_relaxation: float = Field(default=0.5, ge=0, le=1)
    # Constraints
    max_daily_walking_km: float   = Field(default=10.0, ge=0)
    mobility_restricted: bool     = False
    dietary_restrictions: list[str] = Field(default_factory=list)
    # Risk tolerance (0 = very cautious, 1 = adventurous)
    risk_tolerance: float  = Field(default=0.5, ge=0, le=1)
    # Crowd sensitivity (0 = hates crowds, 1 = doesn't care)
    crowd_tolerance: float = Field(default=0.4, ge=0, le=1)


# ─── Itinerary ────────────────────────────────────────────────────────────────

class ItineraryStop(BaseModel):
    node: TravelNode
    arrival_time: datetime
    departure_time: datetime
    transport_to_next: TransportMode | None = None
    notes: str = ""


class DayPlan(BaseModel):
    day: int
    date: datetime
    stops: list[ItineraryStop] = Field(default_factory=list)
    estimated_cost_usd: float = 0.0
    estimated_walking_km: float = 0.0
    fatigue_score: float = 0.0     # 0 = fresh, 1 = exhausted


class Itinerary(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    profile: TravelerProfile
    destination_city: str
    start_date: datetime
    days: list[DayPlan] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    total_distance_km: float = 0.0
    overall_score: float = 0.0     # 0–1 composite quality score
    risk_level: RiskLevel = RiskLevel.LOW
    created_at: datetime = Field(default_factory=datetime.utcnow)
    ai_narrative: str = ""         # LLM-generated explanation


# ─── Optimization ─────────────────────────────────────────────────────────────

class OptimizationWeights(BaseModel):
    """Trade-off weights for multi-objective optimization (must sum ≈ 1)."""
    experience: float = 0.35
    cost: float       = 0.25
    time: float       = 0.20
    crowd: float      = 0.10
    fatigue: float    = 0.10


class OptimizationResult(BaseModel):
    itinerary: Itinerary
    pareto_solutions: list[dict[str, float]] = Field(default_factory=list)
    optimization_time_seconds: float = 0.0
    algorithm: str = "NSGA-II"
    convergence_score: float = 0.0


# ─── Simulation ───────────────────────────────────────────────────────────────

class RiskScenario(BaseModel):
    name: str
    probability: float = Field(ge=0, le=1)
    impact_hours: float = 0.0
    impact_cost_usd: float = 0.0
    event_type: EventType


class SimulationResult(BaseModel):
    itinerary_id: UUID
    n_iterations: int
    success_rate: float
    avg_delay_hours: float
    avg_extra_cost_usd: float
    worst_case_delay_hours: float
    risk_hotspots: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW


# ─── Replanning ───────────────────────────────────────────────────────────────

class DisruptionEvent(BaseModel):
    event_type: EventType
    affected_node_id: str | None = None
    occurred_at: datetime = Field(default_factory=datetime.utcnow)
    severity: float = Field(default=0.5, ge=0, le=1)
    details: dict[str, Any] = Field(default_factory=dict)


class ReplanResult(BaseModel):
    original_itinerary_id: UUID
    new_itinerary: Itinerary
    trigger_event: DisruptionEvent
    changes_summary: list[str] = Field(default_factory=list)
    replan_time_ms: float = 0.0
