"""
Trip AI Platform -- Main Entry Point
=====================================
Run modes:
  python main.py demo          -- CLI demo: full plan + simulation + replan
  python main.py api           -- Start FastAPI server
  python main.py dashboard     -- Start Streamlit dashboard
  python main.py rl-train      -- Train the RL agent
"""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime
from pathlib import Path

# Fix Windows terminal Unicode issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def load_graph():
    from trip_ai.graph_engine import TravelGraph
    g = TravelGraph()
    data_path = Path("data/sample_data.json")
    if data_path.exists():
        with open(data_path) as f:
            g.load_from_dict(json.load(f))
    print(f"Graph loaded: {g}")
    return g


# ─── CLI Demo ─────────────────────────────────────────────────────────────────

def run_demo():
    print("\n" + "="*60)
    print("  TRIP AI PLATFORM — FULL PIPELINE DEMO")
    print("="*60)

    from trip_ai.core.models import (
        OptimizationWeights,
        TravelerProfile,
        EventType,
    )
    from trip_ai.ai_planner import HybridTripPlanner
    from trip_ai.simulation import MonteCarloSimulator, CrowdPredictor
    from trip_ai.simulation.pricing_engine import DynamicPricingEngine, PriceCategory
    from trip_ai.replanning import ReplanEngine, DisruptionEventHandler
    from trip_ai.core.models import DisruptionEvent

    graph = load_graph()

    # ── 1. Build traveller profile ─────────────────────────────────────────
    print("\n[1] Building traveller profile...")
    profile = TravelerProfile(
        name="Alex",
        budget_usd=1500,
        trip_duration_days=4,
        pref_culture=0.8,
        pref_nature=0.5,
        pref_food=0.7,
        pref_adventure=0.3,
        pref_relaxation=0.4,
        crowd_tolerance=0.3,
    )
    print(f"    Traveller: {profile.name} | Budget: ${profile.budget_usd} | {profile.trip_duration_days} days")

    # ── 2. Update crowds with real-time prediction ─────────────────────────
    print("\n[2] Predicting current crowd levels...")
    crowd_pred = CrowdPredictor()
    attractions = graph.nodes_in_city("Tokyo")
    now = datetime.utcnow()
    updated = crowd_pred.update_graph_crowds(attractions, now)
    for node in updated[:3]:
        graph.update_node_attr(node.id, current_crowd_level=node.current_crowd_level)
    print(f"    Updated crowd levels for {len(updated)} nodes")

    # ── 3. Run hybrid planner (NSGA-II + Claude) ───────────────────────────
    print("\n[3] Running Hybrid Planner (NSGA-II + AI)...")
    weights = OptimizationWeights(experience=0.4, cost=0.3, time=0.15, crowd=0.1, fatigue=0.05)
    planner = HybridTripPlanner(graph)

    try:
        result = planner.plan(
            profile=profile,
            city="Tokyo",
            start_date=datetime(2026, 3, 15),
            weights=weights,
            use_llm_draft=True,
        )
        itinerary = result.itinerary
        print(f"    Optimised in {result.optimization_time_seconds:.1f}s")
        print(f"    Pareto solutions found: {len(result.pareto_solutions)}")
        print(f"    Overall score: {itinerary.overall_score:.0%}")
        print(f"    Total cost: ${itinerary.total_cost_usd:.0f}")
        print(f"    Days planned: {len(itinerary.days)}")

        print("\n    ITINERARY:")
        for day in itinerary.days:
            stops = " -> ".join(s.node.name for s in day.stops) or "Rest day"
            print(f"    Day {day.day}: {stops}")
            print(f"           Cost: ${day.estimated_cost_usd:.0f} | Fatigue: {day.fatigue_score:.0%}")

        if itinerary.ai_narrative:
            print(f"\n    AI Narrative (excerpt):")
            print(f"    {itinerary.ai_narrative[:400]}...")

    except Exception as e:
        print(f"    [Planner error - is ANTHROPIC_API_KEY set?] {e}")
        print("    Running without LLM draft...")
        result = planner.plan(
            profile=profile,
            city="Tokyo",
            start_date=datetime(2026, 3, 15),
            weights=weights,
            use_llm_draft=False,
        )
        itinerary = result.itinerary
        print(f"    Optimised in {result.optimization_time_seconds:.1f}s | Score: {itinerary.overall_score:.0%}")
        for day in itinerary.days:
            stops = " -> ".join(s.node.name for s in day.stops) or "Rest day"
            print(f"    Day {day.day}: {stops}")

    # ── 4. Monte Carlo risk simulation ────────────────────────────────────
    print("\n[4] Running Monte Carlo Risk Simulation (10,000 trials)...")
    sim = MonteCarloSimulator(n_iterations=10_000, seed=42)
    risk = sim.simulate(itinerary)
    print(f"    Success rate:   {risk.success_rate:.1%}")
    print(f"    Avg delay:      {risk.avg_delay_hours:.1f}h")
    print(f"    Avg extra cost: ${risk.avg_extra_cost_usd:.0f}")
    print(f"    Worst case:     {risk.worst_case_delay_hours:.1f}h delay")
    print(f"    Risk level:     {risk.risk_level.value.upper()}")
    print(f"    Top hotspots:   {', '.join(risk.risk_hotspots)}")

    # ── 5. Price forecast ─────────────────────────────────────────────────
    print("\n[5] Dynamic Pricing Forecast...")
    pricing = DynamicPricingEngine()
    from datetime import date
    trip_date = date(2026, 3, 15)
    flight_window = pricing.optimal_booking_window(600, trip_date, PriceCategory.FLIGHT)
    hotel_window  = pricing.optimal_booking_window(150, trip_date, PriceCategory.HOTEL)
    print(f"    [Flight] Book on {flight_window.get('best_booking_date')} -> ${flight_window.get('predicted_best_price_usd', 0):.0f} (save {flight_window.get('savings_pct', 0)}%)")
    print(f"    [Hotel]  Book on {hotel_window.get('best_booking_date')} -> ${hotel_window.get('predicted_best_price_usd', 0):.0f} (save {hotel_window.get('savings_pct', 0)}%)")

    budget = pricing.budget_forecast(600, 150, 200, trip_date, profile.trip_duration_days)
    print(f"    [Budget] Full trip estimate: ${budget['total_usd']:.0f} (flight+hotel+attractions+food+transport)")

    # ── 6. Real-time replanning ───────────────────────────────────────────
    print("\n[6] Simulating Disruption & Real-Time Replanning...")
    replan_engine = ReplanEngine(graph)
    ev_handler = DisruptionEventHandler()

    event = ev_handler.create_flight_delay(delay_hours=3.5, flight_node_id="tokyo_airport")
    replan_result = replan_engine.replan(itinerary, event, current_day=1)

    print(f"    Event: {event.event_type.value} (severity {event.severity:.0%})")
    print(f"    Replanned in {replan_result.replan_time_ms:.1f}ms")
    print("    Changes:")
    for change in replan_result.changes_summary:
        print(f"      - {change}")

    # ── 7. Graph stats ────────────────────────────────────────────────────
    print("\n[7] Graph Engine Stats...")
    print(f"    Nodes: {graph.node_count}")
    print(f"    Edges: {graph.edge_count}")

    attractions_ranked = graph.attractions_ranked("Tokyo", profile_tags=["culture", "food"])
    print(f"    Top 3 ranked attractions for this profile:")
    for node in attractions_ranked[:3]:
        print(f"      {node.name} - rating {node.rating}, crowd {node.current_crowd_level:.0%}")

    print("\n" + "="*60)
    print("  DEMO COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  python main.py api        -- Start REST API (port 8000)")
    print("  python main.py dashboard  -- Start Streamlit UI")
    print("  python main.py rl-train   -- Train RL agent\n")


# ─── API server ───────────────────────────────────────────────────────────────

def run_api():
    import uvicorn
    from trip_ai.core.config import settings
    print(f"\nStarting FastAPI server on {settings.api_host}:{settings.api_port}")
    print("Docs: http://localhost:8000/docs\n")
    uvicorn.run(
        "trip_ai.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


# ─── Dashboard ────────────────────────────────────────────────────────────────

def run_dashboard():
    import subprocess
    print("\nStarting Streamlit dashboard...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "trip_ai/dashboard/app.py",
        "--server.port", "8501",
    ])


# ─── RL Training ──────────────────────────────────────────────────────────────

def run_rl_train():
    from trip_ai.core.models import TravelerProfile
    from trip_ai.rl_agent import RLTripAgent

    graph = load_graph()
    profile = TravelerProfile(
        name="RL Agent",
        budget_usd=1500,
        trip_duration_days=4,
        pref_culture=0.7,
        pref_food=0.6,
    )
    attractions = graph.nodes_in_city("Tokyo")
    print(f"\nTraining RL agent on {len(attractions)} Tokyo attractions...")

    agent = RLTripAgent(attractions, profile, graph, model_dir="models/rl")
    agent.train(total_timesteps=50_000, save_path="models/rl/trip_ppo", verbose=1)

    print("\nEvaluating agent...")
    stats = agent.evaluate(n_episodes=10)
    print(f"Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")

    route = agent.plan()
    print(f"\nAgent planned route ({len(route)} attractions):")
    id_to_name = {n.id: n.name for n in attractions}
    for aid in route:
        print(f"  → {id_to_name.get(aid, aid)}")


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if mode == "demo":
        run_demo()
    elif mode == "api":
        run_api()
    elif mode == "dashboard":
        run_dashboard()
    elif mode == "rl-train":
        run_rl_train()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python main.py [demo|api|dashboard|rl-train]")
        sys.exit(1)
