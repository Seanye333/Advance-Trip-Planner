"""
Streamlit Dashboard — Trip AI Platform
========================================
Tabs:
  1. Plan Trip          — Input preferences, run optimizer, view itinerary
  2. Risk Simulation    — Monte Carlo results, risk heatmap
  3. Crowd & Pricing    — Crowd forecast charts, price calendar
  4. Replanning         — Fire disruption events, see live updates
  5. AI Chat            — Conversational trip assistant
  6. RL Agent           — Train/evaluate the RL optimizer

Run with: streamlit run trip_ai/dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trip_ai.core.models import (
    EventType,
    OptimizationWeights,
    TravelerProfile,
)
from trip_ai.graph_engine import TravelGraph
from trip_ai.ai_planner import HybridTripPlanner, LLMPlanner
from trip_ai.simulation import MonteCarloSimulator, CrowdPredictor, DynamicPricingEngine
from trip_ai.simulation.pricing_engine import PriceCategory
from trip_ai.replanning import ReplanEngine, DisruptionEventHandler


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Trip AI Platform",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Load graph (cached) ──────────────────────────────────────────────────────

@st.cache_resource
def load_graph() -> TravelGraph:
    g = TravelGraph()
    data_path = Path(__file__).parent.parent.parent / "data" / "sample_data.json"
    if data_path.exists():
        with open(data_path) as f:
            g.load_from_dict(json.load(f))
    return g


graph = load_graph()
planner = HybridTripPlanner(graph)
simulator = MonteCarloSimulator()
crowd_pred = CrowdPredictor()
pricing = DynamicPricingEngine()
replan_engine = ReplanEngine(graph)
event_handler = DisruptionEventHandler()


# ─── Session state ────────────────────────────────────────────────────────────

if "itinerary" not in st.session_state:
    st.session_state["itinerary"] = None
if "chat_llm" not in st.session_state:
    st.session_state["chat_llm"] = LLMPlanner()
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# ─── Header ───────────────────────────────────────────────────────────────────

st.title("✈️ Trip AI Platform")
st.caption("AI-powered • Multi-objective optimized • Real-time replanning")

tabs = st.tabs([
    "🗺️ Plan Trip",
    "⚠️ Risk Simulation",
    "👥 Crowd & Pricing",
    "🔄 Replanning",
    "💬 AI Chat",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Plan Trip
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.header("Plan Your Trip")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Traveller Profile")
        name         = st.text_input("Name", value="Alex")
        city         = st.text_input("Destination City", value="Tokyo")
        budget       = st.slider("Budget ($)", 500, 10_000, 2_000, 100)
        days         = st.slider("Duration (days)", 1, 14, 5)
        start_date   = st.date_input("Start Date", value=datetime.utcnow().date())

        st.subheader("Preferences (0–10)")
        pref_culture    = st.slider("Culture / History", 0, 10, 7)
        pref_nature     = st.slider("Nature / Parks", 0, 10, 5)
        pref_food       = st.slider("Food & Markets", 0, 10, 6)
        pref_adventure  = st.slider("Adventure / Sport", 0, 10, 4)
        pref_relaxation = st.slider("Relaxation / Spa", 0, 10, 5)

        st.subheader("Optimisation Weights")
        w_exp    = st.slider("Experience", 0.0, 1.0, 0.35, 0.05)
        w_cost   = st.slider("Cost",       0.0, 1.0, 0.25, 0.05)
        w_time   = st.slider("Time",       0.0, 1.0, 0.20, 0.05)
        w_crowd  = st.slider("Crowd",      0.0, 1.0, 0.10, 0.05)
        w_fatigue= st.slider("Fatigue",    0.0, 1.0, 0.10, 0.05)

        use_llm = st.checkbox("Use Claude AI draft reasoning", value=True)
        go = st.button("🚀 Generate Optimal Itinerary", use_container_width=True)

    with col2:
        if go or st.session_state["itinerary"]:
            if go:
                profile = TravelerProfile(
                    name=name,
                    budget_usd=budget,
                    trip_duration_days=days,
                    pref_culture=pref_culture / 10,
                    pref_nature=pref_nature / 10,
                    pref_food=pref_food / 10,
                    pref_adventure=pref_adventure / 10,
                    pref_relaxation=pref_relaxation / 10,
                )
                weights = OptimizationWeights(
                    experience=w_exp, cost=w_cost, time=w_time,
                    crowd=w_crowd, fatigue=w_fatigue,
                )
                with st.spinner("Running NSGA-II optimization + Claude reasoning..."):
                    try:
                        result = planner.plan(
                            profile=profile,
                            city=city,
                            start_date=datetime.combine(start_date, datetime.min.time()),
                            weights=weights,
                            use_llm_draft=use_llm,
                        )
                        st.session_state["itinerary"] = result.itinerary
                        st.session_state["opt_result"] = result
                        st.success(f"Optimised in {result.optimization_time_seconds:.1f}s — {len(result.pareto_solutions)} Pareto solutions found")
                    except Exception as e:
                        st.error(f"Optimisation failed: {e}")

            itinerary = st.session_state.get("itinerary")
            if itinerary:
                # Summary metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Cost", f"${itinerary.total_cost_usd:.0f}")
                m2.metric("Overall Score", f"{itinerary.overall_score:.0%}")
                m3.metric("Risk Level", itinerary.risk_level.value.upper())
                m4.metric("Days Planned", len(itinerary.days))

                # Narrative
                if itinerary.ai_narrative:
                    with st.expander("📖 AI Narrative", expanded=True):
                        st.write(itinerary.ai_narrative)

                # Day-by-day breakdown
                for day in itinerary.days:
                    with st.expander(f"Day {day.day} — {day.date.strftime('%A, %b %d')}"):
                        if not day.stops:
                            st.write("No stops planned.")
                            continue
                        for stop in day.stops:
                            st.markdown(
                                f"**{stop.arrival_time.strftime('%H:%M')}** → "
                                f"**{stop.node.name}** ({stop.node.node_type.value}) | "
                                f"⭐ {stop.node.rating:.1f} | "
                                f"💰 ${stop.node.entry_cost_usd:.0f} | "
                                f"⏱ {stop.node.avg_visit_duration_hours:.1f}h | "
                                f"Crowd: {stop.node.current_crowd_level:.0%}"
                            )
                        st.caption(
                            f"Day cost: ${day.estimated_cost_usd:.0f} | "
                            f"Fatigue: {day.fatigue_score:.0%}"
                        )

                # Pareto front scatter
                result = st.session_state.get("opt_result")
                if result and result.pareto_solutions:
                    st.subheader("Pareto Front (trade-off space)")
                    import pandas as pd
                    df = pd.DataFrame(result.pareto_solutions)
                    if "cost_usd" in df.columns and "experience" in df.columns:
                        fig = px.scatter(
                            df, x="cost_usd", y="experience",
                            color="fatigue", size="time_hours",
                            labels={"cost_usd": "Cost ($)", "experience": "Experience Score"},
                            title="Cost vs Experience — Pareto Frontier",
                            color_continuous_scale="RdYlGn_r",
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Configure your trip profile and click **Generate Optimal Itinerary**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Risk Simulation
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.header("Monte Carlo Risk Simulation")
    itinerary = st.session_state.get("itinerary")
    if not itinerary:
        st.warning("Generate an itinerary first (Plan Trip tab).")
    else:
        n_iter = st.slider("Simulation iterations", 1_000, 50_000, 10_000, 1_000)
        if st.button("▶ Run Simulation"):
            with st.spinner(f"Running {n_iter:,} Monte Carlo trials..."):
                sim = MonteCarloSimulator(n_iterations=n_iter)
                result = sim.simulate(itinerary)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Success Rate", f"{result.success_rate:.1%}")
            c2.metric("Avg Delay", f"{result.avg_delay_hours:.1f}h")
            c3.metric("Avg Extra Cost", f"${result.avg_extra_cost_usd:.0f}")
            c4.metric("Worst Delay", f"{result.worst_case_delay_hours:.1f}h")

            st.subheader("Risk Level")
            color = {"low": "🟢", "medium": "🟡", "high": "🟠", "extreme": "🔴"}
            st.markdown(f"## {color.get(result.risk_level.value, '⚪')} {result.risk_level.value.upper()}")

            st.subheader("Top Risk Hotspots")
            for hotspot in result.risk_hotspots:
                st.markdown(f"- ⚠️ {hotspot}")

            # Sensitivity analysis
            st.subheader("Sensitivity Analysis (which risks hurt most)")
            with st.spinner("Running sensitivity analysis..."):
                sensitivity = sim.sensitivity_analysis(itinerary, n_samples=500)
            import pandas as pd
            df_sens = pd.DataFrame(
                [(k, v) for k, v in sensitivity.items()],
                columns=["Risk Event", "Δ Success Rate"]
            )
            fig_sens = px.bar(
                df_sens, x="Δ Success Rate", y="Risk Event",
                orientation="h",
                title="Impact of doubling each risk probability",
                color="Δ Success Rate", color_continuous_scale="RdYlGn_r",
            )
            st.plotly_chart(fig_sens, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Crowd & Pricing
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.header("Crowd & Dynamic Pricing")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Crowd Forecast")
        city_nodes = graph.nodes_in_city(st.session_state.get("itinerary", type('', (), {'destination_city': 'Tokyo'})()).destination_city if st.session_state.get("itinerary") else "Tokyo")
        if city_nodes:
            selected_node = st.selectbox("Select Attraction", city_nodes, format_func=lambda n: n.name)
            forecast_date = st.date_input("Date", key="crowd_date")
            hours = list(range(8, 21))
            crowds = [crowd_pred.predict(selected_node, datetime.combine(forecast_date, datetime.min.time().replace(hour=h))) for h in hours]
            import pandas as pd
            fig_crowd = px.line(
                x=hours, y=crowds,
                labels={"x": "Hour of Day", "y": "Crowd Level"},
                title=f"Crowd Forecast — {selected_node.name}",
            )
            fig_crowd.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="High Crowd")
            st.plotly_chart(fig_crowd, use_container_width=True)
            best_h = crowd_pred.best_visit_time(selected_node, datetime.combine(forecast_date, datetime.min.time()))
            st.success(f"Best visit time: **{best_h:02d}:00**")
        else:
            st.info("No attractions loaded. Add city data.")

    with col_b:
        st.subheader("Price Calendar")
        cat = st.selectbox("Category", ["Flight", "Hotel"])
        base_price = st.number_input("Base Price ($)", value=500 if cat == "Flight" else 120)
        price_category = PriceCategory.FLIGHT if cat == "Flight" else PriceCategory.HOTEL
        cal = pricing.price_calendar(
            base_price_usd=base_price,
            category=price_category,
            start_date=datetime.utcnow().date(),
            days=30,
        )
        import pandas as pd
        df_cal = pd.DataFrame(cal)
        fig_price = px.bar(
            df_cal, x="date", y="predicted_price_usd",
            color="relative_price",
            labels={"predicted_price_usd": "Predicted Price ($)"},
            title=f"{cat} Price Forecast (next 30 days)",
            color_continuous_scale="RdYlGn_r",
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # Best booking window
        trip_date = st.date_input("Trip Date", value=(datetime.utcnow() + timedelta(days=45)).date(), key="bw_date")
        window = pricing.optimal_booking_window(base_price, trip_date, price_category)
        if "best_booking_days_before" in window:
            st.info(
                f"**Best time to book:** {window['best_booking_date']} "
                f"({window['best_booking_days_before']} days before trip) — "
                f"${window['predicted_best_price_usd']:.0f} "
                f"(save {window['savings_pct']}%)"
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Replanning
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.header("Real-Time Replanning Engine")
    itinerary = st.session_state.get("itinerary")
    if not itinerary:
        st.warning("Generate an itinerary first.")
    else:
        st.markdown("Simulate a real-world disruption and watch the itinerary adapt in real time.")

        event_type = st.selectbox(
            "Disruption Type",
            [e.value for e in EventType],
        )
        severity = st.slider("Severity", 0.0, 1.0, 0.5)
        current_day = st.number_input("Current Day", 1, itinerary.profile.trip_duration_days, 1)
        delay_hours = st.number_input("Delay Hours (for flight delays)", 0.0, 24.0, 2.0)

        if st.button("⚡ Trigger Disruption & Replan"):
            details = {"delay_hours": delay_hours} if event_type == "flight_delay" else {}
            from trip_ai.core.models import DisruptionEvent
            event = DisruptionEvent(
                event_type=EventType(event_type),
                severity=severity,
                details=details,
            )
            with st.spinner("Replanning..."):
                replan_result = replan_engine.replan(itinerary, event, int(current_day))

            st.success(f"Replanned in {replan_result.replan_time_ms:.1f}ms")
            st.subheader("Changes Made")
            for change in replan_result.changes_summary:
                st.markdown(f"• {change}")

            st.subheader("New Itinerary")
            for day in replan_result.new_itinerary.days:
                st.markdown(f"**Day {day.day}:** " + " → ".join(s.node.name for s in day.stops))

            if st.button("Apply New Itinerary"):
                st.session_state["itinerary"] = replan_result.new_itinerary
                st.success("Itinerary updated!")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: AI Chat
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.header("💬 AI Travel Assistant")
    itinerary = st.session_state.get("itinerary")

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask anything about your trip...")
    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply = st.session_state["chat_llm"].chat(user_input, itinerary)
                except Exception as e:
                    reply = f"[Error: {e}]"
            st.write(reply)
            st.session_state["chat_history"].append({"role": "assistant", "content": reply})

    if st.button("🗑️ Clear Chat"):
        st.session_state["chat_history"] = []
        st.session_state["chat_llm"].reset_chat()
        st.rerun()
