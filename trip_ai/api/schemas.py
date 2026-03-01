"""Request/response schemas for the FastAPI layer."""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from trip_ai.core.models import OptimizationWeights, TravelerProfile


class PlanTripRequest(BaseModel):
    profile: TravelerProfile
    city: str
    start_date: datetime = Field(default_factory=datetime.utcnow)
    weights: OptimizationWeights = Field(default_factory=OptimizationWeights)
    use_llm_draft: bool = True


class ChatRequest(BaseModel):
    message: str
    itinerary_id: UUID | None = None


class FeedbackRequest(BaseModel):
    itinerary_id: UUID
    node_id: str
    event: str  # "like" | "dislike" | "skip" | "extended_visit"


class ReplanRequest(BaseModel):
    itinerary_id: UUID
    event_type: str
    affected_node_id: str | None = None
    severity: float = 0.5
    details: dict = Field(default_factory=dict)
    current_day: int = 1


class SimulateRequest(BaseModel):
    itinerary_id: UUID
    n_iterations: int = Field(default=10_000, ge=100, le=100_000)
