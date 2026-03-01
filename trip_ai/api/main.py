"""
FastAPI Application — Trip AI Platform
========================================
Endpoints:
  POST /trips/plan          → Generate optimised itinerary
  GET  /trips/{id}          → Retrieve itinerary
  POST /trips/{id}/replan   → Trigger real-time replanning
  POST /trips/{id}/simulate → Monte Carlo risk simulation
  POST /trips/{id}/chat     → AI travel assistant chat
  POST /trips/{id}/feedback → Submit attraction feedback
  GET  /health              → Health check
"""
from __future__ import annotations

import json
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from trip_ai.ai_planner import HybridTripPlanner, LLMPlanner
from trip_ai.api.schemas import (
    ChatRequest,
    FeedbackRequest,
    PlanTripRequest,
    ReplanRequest,
    SimulateRequest,
)
from trip_ai.core.models import DisruptionEvent, EventType, Itinerary
from trip_ai.graph_engine import TravelGraph
from trip_ai.replanning import DisruptionEventHandler, ReplanEngine
from trip_ai.simulation import MonteCarloSimulator


# ─── App factory ──────────────────────────────────────────────────────────────

def create_app(graph: TravelGraph) -> FastAPI:
    app = FastAPI(
        title="Trip AI Platform",
        description="AI-powered real-time trip planning engine",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Shared services ────────────────────────────────────────────────────
    planner      = HybridTripPlanner(graph)
    llm          = LLMPlanner()
    simulator    = MonteCarloSimulator()
    replan_engine = ReplanEngine(graph)
    event_handler = DisruptionEventHandler()

    # ── In-memory store (replace with Redis / PostgreSQL in production) ────
    _itineraries: dict[str, Itinerary] = {}
    _chat_sessions: dict[str, LLMPlanner] = {}

    # ─── Routes ───────────────────────────────────────────────────────────────

    @app.get("/health")
    def health():
        return {"status": "ok", "graph_nodes": graph.node_count, "graph_edges": graph.edge_count}

    @app.post("/trips/plan")
    def plan_trip(req: PlanTripRequest):
        try:
            result = planner.plan(
                profile=req.profile,
                city=req.city,
                start_date=req.start_date,
                weights=req.weights,
                use_llm_draft=req.use_llm_draft,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        itinerary = result.itinerary
        _itineraries[str(itinerary.id)] = itinerary

        return {
            "itinerary_id": str(itinerary.id),
            "days": len(itinerary.days),
            "total_cost_usd": itinerary.total_cost_usd,
            "overall_score": itinerary.overall_score,
            "risk_level": itinerary.risk_level.value,
            "narrative": itinerary.ai_narrative,
            "optimization_time_s": result.optimization_time_seconds,
            "pareto_solutions_count": len(result.pareto_solutions),
        }

    @app.get("/trips/{itinerary_id}")
    def get_trip(itinerary_id: str):
        itinerary = _itineraries.get(itinerary_id)
        if not itinerary:
            raise HTTPException(status_code=404, detail="Itinerary not found")
        return itinerary.model_dump(mode="json")

    @app.post("/trips/{itinerary_id}/replan")
    def replan_trip(itinerary_id: str, req: ReplanRequest):
        itinerary = _itineraries.get(itinerary_id)
        if not itinerary:
            raise HTTPException(status_code=404, detail="Itinerary not found")

        event = DisruptionEvent(
            event_type=EventType(req.event_type),
            affected_node_id=req.affected_node_id,
            severity=req.severity,
            details=req.details,
        )
        result = replan_engine.replan(itinerary, event, req.current_day)
        new_id = str(result.new_itinerary.id)
        _itineraries[new_id] = result.new_itinerary

        return {
            "new_itinerary_id": new_id,
            "changes": result.changes_summary,
            "replan_time_ms": result.replan_time_ms,
        }

    @app.post("/trips/{itinerary_id}/simulate")
    def simulate_trip(itinerary_id: str, req: SimulateRequest):
        itinerary = _itineraries.get(itinerary_id)
        if not itinerary:
            raise HTTPException(status_code=404, detail="Itinerary not found")

        sim = MonteCarloSimulator(n_iterations=req.n_iterations)
        result = sim.simulate(itinerary)
        return result.model_dump()

    @app.post("/trips/{itinerary_id}/chat")
    def chat(itinerary_id: str, req: ChatRequest):
        itinerary = _itineraries.get(itinerary_id)
        session_llm = _chat_sessions.setdefault(itinerary_id, LLMPlanner())
        reply = session_llm.chat(req.message, itinerary)
        return {"reply": reply}

    @app.post("/trips/{itinerary_id}/chat/stream")
    def chat_stream(itinerary_id: str, req: ChatRequest):
        session_llm = _chat_sessions.setdefault(itinerary_id, LLMPlanner())

        def generator():
            for chunk in session_llm.chat_stream(req.message):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generator(), media_type="text/event-stream")

    @app.post("/trips/{itinerary_id}/feedback")
    def record_feedback(itinerary_id: str, req: FeedbackRequest):
        # In production: update preference model in DB
        return {"status": "feedback recorded", "node_id": req.node_id, "event": req.event}

    return app


# ─── Entry point (used by uvicorn) ────────────────────────────────────────────
# In production, inject a real graph loaded from DB.
# For demo: build from sample data.

def _build_demo_graph() -> TravelGraph:
    import json, pathlib
    g = TravelGraph()
    data_path = pathlib.Path(__file__).parent.parent.parent / "data" / "sample_data.json"
    if data_path.exists():
        with open(data_path) as f:
            g.load_from_dict(json.load(f))
    return g


graph = _build_demo_graph()
app = create_app(graph)
