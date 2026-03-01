# ✈️ Trip AI Platform — Enterprise-Grade AI Travel Intelligence System

> **Not a travel app. A real-time optimization engine that plans, simulates, and continuously adapts trips like Google Maps adapts routes.**

---

## 🏗️ Architecture

```
User Preferences
       │
       ▼
┌──────────────────┐     ┌─────────────────────┐
│  Preference      │────▶│  LLM Draft Planner  │ (Claude claude-sonnet-4-6)
│  Engine          │     │  (Reasoning Pass)    │
└──────────────────┘     └──────────┬──────────┘
                                     │ draft
                                     ▼
┌──────────────────┐     ┌─────────────────────┐
│  Travel Graph    │────▶│  NSGA-II Optimizer  │ (Multi-objective)
│  Engine (NX)     │     │  5 objectives       │
└──────────────────┘     └──────────┬──────────┘
                                     │ optimal set
                                     ▼
                          ┌─────────────────────┐
                          │  Simulated Annealing│ (TSP route order)
                          └──────────┬──────────┘
                                     │ ordered itinerary
                                     ▼
┌──────────────────┐     ┌─────────────────────┐
│  Monte Carlo     │────▶│  Risk Simulation    │ (10K trials)
│  Simulator       │     │  + Sensitivity      │
└──────────────────┘     └──────────┬──────────┘
                                     │ risk-annotated itinerary
                                     ▼
┌──────────────────┐     ┌─────────────────────┐
│  Disruption      │────▶│  Replan Engine      │ (< 50ms response)
│  Event Handler   │     │  Google Maps-style  │
└──────────────────┘     └──────────┬──────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                  ▼
           ┌──────────────┐                  ┌──────────────┐
           │  FastAPI     │                  │  Streamlit   │
           │  REST API    │                  │  Dashboard   │
           └──────────────┘                  └──────────────┘
```

---

## 🧠 What Makes This Advanced

| Feature | Technology | Status |
|---------|-----------|--------|
| Multi-objective optimization | NSGA-II (pymoo) | ✅ |
| AI reasoning + explanation | Claude claude-sonnet-4-6 (Anthropic) | ✅ |
| TSP route optimization | Simulated Annealing | ✅ |
| MILP daily scheduling | PuLP (CBC solver) | ✅ |
| Multi-path routing | Label-setting Pareto algorithm | ✅ |
| Risk quantification | Monte Carlo (10K trials) | ✅ |
| Crowd forecasting | Rule-based + ML-ready predictor | ✅ |
| Dynamic pricing | Additive decomposition model | ✅ |
| Real-time replanning | Event-driven engine (<50ms) | ✅ |
| Reinforcement learning | PPO via Stable-Baselines3 | ✅ |
| Preference learning | EMA-based feedback loop | ✅ |
| REST API | FastAPI + streaming SSE | ✅ |
| Interactive dashboard | Streamlit + Plotly | ✅ |

---

## 📂 Project Structure

```
trip_ai/
├── core/
│   ├── config.py           ← Environment config (pydantic-settings)
│   └── models.py           ← All data models (single source of truth)
│
├── graph_engine/
│   ├── travel_graph.py     ← NetworkX multigraph of the travel world
│   └── pathfinder.py       ← Multi-objective Pareto pathfinding
│
├── optimization/
│   ├── multi_objective.py  ← NSGA-II trip optimizer (5 objectives)
│   ├── itinerary_solver.py ← MILP daily scheduler (PuLP)
│   └── simulated_annealing.py ← 2-opt TSP route optimizer
│
├── ai_planner/
│   ├── llm_planner.py      ← Claude API: draft, explain, chat, stream
│   ├── preference_engine.py← EMA preference learning from feedback
│   └── hybrid_planner.py   ← Full pipeline orchestrator
│
├── simulation/
│   ├── monte_carlo.py      ← 10K-trial risk simulator + sensitivity
│   ├── crowd_predictor.py  ← Time/season/event crowd forecasting
│   └── pricing_engine.py   ← Flight/hotel/attraction price prediction
│
├── rl_agent/
│   ├── trip_env.py         ← Gymnasium MDP environment
│   └── agent.py            ← PPO agent (Stable-Baselines3 wrapper)
│
├── replanning/
│   ├── event_handler.py    ← Disruption event factory + classification
│   └── replan_engine.py    ← Real-time itinerary adaptation (<50ms)
│
├── api/
│   ├── main.py             ← FastAPI app + all routes
│   └── schemas.py          ← Request/response Pydantic models
│
└── dashboard/
    └── app.py              ← Streamlit 5-tab interactive dashboard

data/
└── sample_data.json        ← 15 Tokyo nodes + 19 edges (ready to run)

tests/
├── test_graph_engine.py
├── test_optimization.py
├── test_simulation.py
└── test_replanning.py
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run the CLI demo

```bash
python main.py demo
```

This runs the **full pipeline**:
- Loads Tokyo travel graph
- Predicts real-time crowd levels
- Runs NSGA-II optimization + Claude AI reasoning
- Executes 10,000 Monte Carlo risk trials
- Forecasts flight & hotel prices
- Simulates a flight delay → replans in milliseconds

### 4. Start the dashboard

```bash
python main.py dashboard
# → http://localhost:8501
```

### 5. Start the REST API

```bash
python main.py api
# → http://localhost:8000/docs
```

### 6. Train the RL agent

```bash
python main.py rl-train
```

### 7. Run tests

```bash
pytest tests/ -v
```

---

## 🔌 REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/trips/plan` | Generate optimised itinerary |
| `GET`  | `/trips/{id}` | Retrieve itinerary |
| `POST` | `/trips/{id}/replan` | Trigger replanning event |
| `POST` | `/trips/{id}/simulate` | Monte Carlo risk simulation |
| `POST` | `/trips/{id}/chat` | AI chat (stateful) |
| `POST` | `/trips/{id}/chat/stream` | Streaming AI chat (SSE) |
| `POST` | `/trips/{id}/feedback` | Submit attraction feedback |
| `GET`  | `/health` | Health check |

### Example: Plan a trip

```bash
curl -X POST http://localhost:8000/trips/plan \
  -H "Content-Type: application/json" \
  -d '{
    "profile": {
      "name": "Alex",
      "budget_usd": 1500,
      "trip_duration_days": 4,
      "pref_culture": 0.8,
      "pref_food": 0.7,
      "pref_nature": 0.5
    },
    "city": "Tokyo",
    "weights": {
      "experience": 0.4,
      "cost": 0.3,
      "time": 0.15,
      "crowd": 0.1,
      "fatigue": 0.05
    }
  }'
```

### Example: Replan after a disruption

```bash
curl -X POST http://localhost:8000/trips/{id}/replan \
  -H "Content-Type: application/json" \
  -d '{
    "itinerary_id": "...",
    "event_type": "flight_delay",
    "severity": 0.6,
    "details": {"delay_hours": 3.5},
    "current_day": 1
  }'
```

---

## 🎯 Optimization Objectives

The NSGA-II optimizer simultaneously handles **5 objectives**:

```
Maximise: experience_score
           = rating × weather_comfort × preference_match × (1 + tag_boost)

Minimise: total_cost_usd
           = Σ entry_costs + travel_costs

Minimise: total_time_hours
           = Σ visit_durations + travel_times

Minimise: avg_crowd_level
           = Σ crowd_levels / n_stops

Minimise: fatigue_score
           = Σ (visit_duration × 0.1 + crowd × 0.15)
```

The **Pareto frontier** of non-dominated solutions is returned. A **weighted Tchebycheff scalarisation** selects the final plan based on the user's trade-off weights.

---

## ⚠️ Risk Simulation

The Monte Carlo engine simulates **8 disruption scenarios per trial**:

| Scenario | Default Prob | Time Lost | Extra Cost |
|----------|-------------|-----------|------------|
| Short flight delay | 20% | 1–3h | $0–50 |
| Missed connection | 5% | 4–12h | $100–500 |
| Attraction closure | 10% | 0.5–2h | $0–30 |
| Severe weather | 15% | 1–4h | $20–100 |
| Traffic spike | 25% | 0.3–1h | $0 |
| Hotel overbooking | 3% | 2–4h | $50–200 |
| Illness/fatigue | 8% | 2–8h | $20–80 |
| Strike | 2% | 4–24h | $100–300 |

**Risk Level Classification:**
- 🟢 LOW: success rate ≥ 90%, worst delay < 6h
- 🟡 MEDIUM: ≥ 75%, < 12h
- 🟠 HIGH: ≥ 55%
- 🔴 EXTREME: < 55%

---

## 🔄 Real-Time Replanning

Each disruption event type triggers a specific strategy:

| Event | Strategy | Latency |
|-------|----------|---------|
| `flight_delay` | Shift all timings; drop last stop if >4h | <5ms |
| `attraction_closure` | Remove node; insert best alternative | <10ms |
| `weather_change` | Swap outdoor → indoor attractions | <10ms |
| `fatigue` | Trim stops proportional to energy level | <5ms |
| `budget_exceeded` | Remove most expensive remaining stop | <5ms |
| `traffic_spike` | Add buffer time to current-day schedule | <5ms |
| `strike` | Flag all transit legs; request full replan | <5ms |

---

## 🤖 Reinforcement Learning Agent

The RL agent models trip planning as an MDP:

**State** (8-D continuous):
```
[budget_remaining, time_remaining, energy,
 crowd_level, weather_score, visited_ratio,
 current_lat, current_lon]
```

**Actions** (discrete):
- `0..N-1` → visit attraction i
- `N` → rest (recover 0.15 energy)
- `N+1` → end day early

**Reward**:
```
r = rating × preference × weather
  - cost/budget
  - crowd × (1 - crowd_tolerance)
  - travel_time × 0.1
  - fatigue × 0.2
  + 2.0 (day completion bonus if ≥ 3 stops)
```

Train with PPO for ~200K steps to get a useful policy.

---

## 🗺️ Build Roadmap

### ✅ Phase 1 — Core Intelligence (DONE)
- Travel graph engine (NetworkX)
- Multi-objective optimizer (NSGA-II)
- AI planner (Claude API)
- Preference learning (EMA)

### ✅ Phase 2 — Simulation & Prediction (DONE)
- Monte Carlo risk simulator
- Crowd & congestion predictor
- Dynamic pricing engine

### ✅ Phase 3 — Real-Time Adaptation (DONE)
- Disruption event handler
- Real-time replanning engine

### ✅ Phase 4 — RL & Full Stack (DONE)
- Gymnasium environment
- PPO agent (Stable-Baselines3)
- FastAPI REST backend
- Streamlit dashboard

### 🔜 Phase 5 — Production Hardening
- [ ] PostgreSQL persistence layer
- [ ] Redis event bus for real-time updates
- [ ] Mapbox integration for real routing
- [ ] Google Popular Times API for crowd data
- [ ] Skyscanner/Amadeus API for live pricing
- [ ] Auth (JWT) + multi-user support
- [ ] Docker Compose deployment
- [ ] CI/CD pipeline

### 🔜 Phase 6 — Advanced Features
- [ ] Group travel optimization (multi-person MDP)
- [ ] Accessibility routing (wheelchair, elderly)
- [ ] Visa & entry requirement integration
- [ ] Health risk scoring (CDC travel advisories)
- [ ] Carbon footprint optimization
- [ ] Offline mode with cached graph

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| AI/LLM | Anthropic Claude claude-sonnet-4-6 |
| Multi-objective opt. | pymoo (NSGA-II, NSGA-III) |
| MILP scheduling | PuLP (CBC solver) |
| Graph algorithms | NetworkX |
| Reinforcement learning | Stable-Baselines3 + Gymnasium |
| ML/Prediction | scikit-learn, XGBoost |
| Backend API | FastAPI + uvicorn |
| Dashboard | Streamlit + Plotly |
| Data models | Pydantic v2 |
| Config | pydantic-settings |

---

## 💡 Key Engineering Concepts Demonstrated

1. **Multi-objective optimization** — Pareto frontier, Tchebycheff scalarisation, NSGA-II
2. **Graph algorithms** — Dijkstra, multi-objective label-setting, TSP/2-opt
3. **LLM integration** — Hybrid AI+math, streaming, structured output, conversation memory
4. **Risk engineering** — Monte Carlo simulation, sensitivity analysis, scenario modelling
5. **Real-time systems** — Event-driven architecture, sub-50ms replanning
6. **Reinforcement learning** — Custom Gymnasium MDP, reward shaping, PPO
7. **Full-stack architecture** — FastAPI, SSE streaming, Streamlit, Pydantic v2

---

*Built with Claude claude-sonnet-4-6 + Python optimization stack*
# Advance-Trip-Planner
