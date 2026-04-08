# 🏠 Build My Home — AI-Powered Building Design System

> **Generate complete building designs from a single sentence.**  
> Powered by NLP, Reinforcement Learning, Structural Engineering, and CPM Scheduling.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Pipeline Flow](#3-pipeline-flow)
4. [Local Setup (Without Docker)](#4-local-setup-without-docker)
5. [Running the Pipeline Manually](#5-running-the-pipeline-manually)
6. [Backend Guide](#6-backend-guide)
7. [Frontend Guide](#7-frontend-guide)
8. [Docker Setup](#8-docker-setup)
9. [Async Job Execution](#9-async-job-execution)
10. [Environment Variables](#10-environment-variables)
11. [Testing](#11-testing)
12. [Troubleshooting](#12-troubleshooting)
13. [Sample Output](#13-sample-output)

---

## 1. Project Overview

**Build My Home** is an end-to-end AI system that converts a plain-English building requirement into a fully designed building — complete with floor plans, structural grids, MEP routing, construction schedules, and cost estimates.

### What it does

Type something like:

> *"I want a 3-floor house with 4 bedrooms, 3 bathrooms, a kitchen, living room, and parking on a 40×60 plot."*

And the system produces:

- ✅ Validated room constraints (IRC / NBC standards)
- ✅ RL-generated floor plan layout for every floor
- ✅ Structural column, beam, and slab grid (ML-predicted spacing)
- ✅ Plumbing and electrical routes (A\* / Dijkstra pathfinding)
- ✅ 41-task construction DAG with dependencies
- ✅ CPM / PERT project timeline and Gantt chart
- ✅ Material, labor, and MEP cost estimate
- ✅ Interactive 3D visualization

### Who is it for?

- Architects and civil engineers for rapid concept design
- Construction companies for early-stage estimation
- Developers building BIM tooling on top of this API

---

## 2. Project Structure

```
build-my-home/
│
├── backend/                        # FastAPI application
│   ├── app/
│   │   ├── main.py                 # FastAPI entry point
│   │   ├── routers/
│   │   │   ├── generate.py         # POST /generate
│   │   │   ├── mep.py              # POST /mep
│   │   │   ├── cost.py             # POST /cost
│   │   │   └── health.py           # GET  /health
│   │   ├── jobs/
│   │   │   ├── celery_app.py       # Celery app instance
│   │   │   └── tasks.py            # Async pipeline tasks
│   │   └── schemas/
│   │       ├── request.py          # Pydantic request models
│   │       └── response.py         # Pydantic response models
│   └── requirements.txt
│
├── ai_models/                      # All AI/ML modules
│   ├── building_nlp.py             # Stage 1 — NLP parser
│   ├── constraint_validator.py     # Stage 2 — Constraint engine
│   ├── floor_plan_env.py           # Stage 3 — RL environment
│   ├── ppo_agent.py                # Stage 3 — PPO agent
│   ├── multifloor_env.py           # Stage 4 — Multi-floor planner
│   ├── multifloor_train.py         # Stage 4 — Hierarchical training
│   ├── structural_constraints.py   # Stage 5 — Vertical alignment
│   ├── structural_grid.py          # Stage 5 — Grid generator
│   ├── load_estimator.py           # Stage 5 — Load distribution
│   ├── column_predictor.py         # Stage 5 — ML spacing model
│   ├── mep_routing.py              # Stage 6 — MEP routing
│   ├── task_engine.py              # Stage 7 — Construction tasks
│   ├── scheduler.py                # Stage 8 — CPM/PERT
│   ├── cost_estimator.py           # Stage 9 — Cost model
│   ├── pipeline.py                 # Orchestrator
│   └── main_pipeline.py            # CLI entry point
│
├── frontend/                       # React application
│   ├── src/
│   │   ├── components/
│   │   │   ├── InputForm.jsx       # Requirement text input
│   │   │   ├── FloorPlanCanvas.jsx # SVG floor plan renderer
│   │   │   ├── StructuralView.jsx  # Column/beam overlay
│   │   │   ├── MEPView.jsx         # Plumbing/electrical routes
│   │   │   ├── GanttChart.jsx      # Schedule visualization
│   │   │   ├── CostDashboard.jsx   # Cost breakdown charts
│   │   │   └── Viewer3D.jsx        # Three.js 3D viewer
│   │   ├── api/
│   │   │   └── client.js           # Axios API client
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf
│
├── docker-compose.yml
├── docker-compose.prod.yml
├── .env.example
└── README.md
```

---

## 3. Pipeline Flow

Each user request passes through 10 sequential stages. Each stage's output is the next stage's input.

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INPUT                           │
│   "3-floor house, 4 beds, 3 baths, kitchen, 40×60 plot"    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────┐
              │  Stage 1 — NLP Parser      │  building_nlp.py
              │  Text → Structured JSON    │  (rule-based + spaCy)
              └──────────────┬─────────────┘
                             │
                             ▼
              ┌────────────────────────────┐
              │  Stage 2 — Constraints     │  constraint_validator.py
              │  Validate IRC/NBC rules    │  (rule engine)
              └──────────────┬─────────────┘
                             │
                             ▼
              ┌────────────────────────────┐
              │  Stage 3 — Floor Plans     │  floor_plan_env.py
              │  RL agent places rooms     │  ppo_agent.py (PPO)
              └──────────────┬─────────────┘
                             │
                             ▼
              ┌────────────────────────────┐
              │  Stage 4 — Multi-Floor     │  multifloor_env.py
              │  L1 → L2 hierarchical PPO  │  (2 specialised agents)
              └──────────────┬─────────────┘
                             │
                             ▼
              ┌────────────────────────────┐
              │  Stage 5 — Structure       │  structural_grid.py
              │  Columns, beams, slabs     │  (GBR ML + IS 456)
              └──────────────┬─────────────┘
                             │
                             ▼
              ┌────────────────────────────┐
              │  Stage 6 — MEP Routing     │  mep_routing.py
              │  Pipes + wiring paths      │  (NetworkX A* / Dijkstra)
              └──────────────┬─────────────┘
                             │
                             ▼
              ┌────────────────────────────┐
              │  Stage 7 — Tasks           │  task_engine.py
              │  41-task construction DAG  │  (networkx DiGraph)
              └──────────────┬─────────────┘
                             │
                             ▼
              ┌────────────────────────────┐
              │  Stage 8 — Schedule        │  scheduler.py
              │  CPM / PERT timeline       │  (forward + backward pass)
              └──────────────┬─────────────┘
                             │
                             ▼
              ┌────────────────────────────┐
              │  Stage 9 — Cost            │  cost_estimator.py
              │  Material / Labor / MEP    │  (GBR × 3 ML models)
              └──────────────┬─────────────┘
                             │
                             ▼
              ┌────────────────────────────┐
              │  Stage 10 — Visualization  │  Three.js / matplotlib
              │  3D model + floor plan PNG │
              └──────────────┬─────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                        JSON RESPONSE                        │
│  layout, structure, mep, tasks, schedule, cost, 3d_model   │
└─────────────────────────────────────────────────────────────┘
```

### Which stages require pre-training?

| Stage | Model | Training required? | How |
|---|---|---|---|
| NLP Parser | Rule-based regex | ❌ No | Runs immediately |
| Constraint Validator | Rule engine | ❌ No | Runs immediately |
| RL Floor Planner (L1) | PPO agent | ✅ Yes — once | `python ai_models/multifloor_train.py` |
| RL Floor Planner (L2) | PPO agent | ✅ Yes — once | Same script |
| Structural Grid | GBR (sklearn) | ⚡ Auto | Trains on startup in ~2s |
| MEP Router | NetworkX graph | ❌ No | Runs immediately |
| Task Engine | DAG logic | ❌ No | Runs immediately |
| Scheduler | CPM math | ❌ No | Runs immediately |
| Cost Estimator | GBR × 3 (sklearn) | ⚡ Auto | Trains on startup in ~3s |
| 3D Visualizer | Three.js / matplotlib | ❌ No | Renders on demand |

> **Summary:** You only need to train the two RL agents once, and only if you want ML-generated floor plans. All other models train automatically on first startup.

---

## 4. Local Setup (Without Docker)

### Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- Git

### 4.1 — Clone the repository

```bash
git clone https://github.com/your-username/build-my-home.git
cd build-my-home
```

### 4.2 — Set up the Python environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate

# Install all Python dependencies
pip install -r backend/requirements.txt
```

### 4.3 — Install AI model dependencies

```bash
pip install \
  networkx \
  scikit-learn \
  matplotlib \
  pandas \
  numpy \
  fastapi \
  uvicorn[standard] \
  celery \
  redis \
  python-dotenv \
  pydantic
```

> **Optional (for transformer-based NLP):**
> ```bash
> pip install spacy transformers torch
> python -m spacy download en_core_web_sm
> ```

### 4.4 — Configure environment variables

```bash
cp .env.example .env
# Edit .env with your values (see Section 10)
```

### 4.5 — Train the RL agents (one-time)

This step is only required if you want AI-generated floor plans instead of the deterministic fallback planner.

```bash
# Train both Level-1 (ground floor) and Level-2 (upper floor) PPO agents
# Expected time: 5–10 minutes on CPU for 300 episodes

python ai_models/multifloor_train.py --episodes 300 --floors 3

# Trained weights are saved to:
# ai_models/outputs/l1_agent.pkl
# ai_models/outputs/l2_agent.pkl
```

### 4.6 — Start Redis (required for async jobs)

```bash
# macOS
brew install redis && brew services start redis

# Ubuntu / Debian
sudo apt install redis-server && sudo systemctl start redis

# Verify Redis is running
redis-cli ping
# Expected output: PONG
```

### 4.7 — Start the Celery worker

Open a new terminal:

```bash
source venv/bin/activate
cd backend
celery -A app.jobs.celery_app worker --loglevel=info --concurrency=4
```

### 4.8 — Start the FastAPI backend

Open another new terminal:

```bash
source venv/bin/activate
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Visit `http://localhost:8000/docs` for the auto-generated API documentation.

### 4.9 — Start the React frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

You should see:

```
  VITE v5.x  ready in 300ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://192.168.x.x:5173/
```

Open your browser at `http://localhost:5173`.

---

## 5. Running the Pipeline Manually

You can run the full pipeline from the command line without the API server — useful for testing, batch processing, or debugging individual stages.

### Run the full pipeline

```bash
cd ai_models

python main_pipeline.py \
  --input "I want a 3 floor house with 4 bedrooms, 3 bathrooms, kitchen, living room, parking and a garden on a 40x60 plot." \
  --floors 3 \
  --output outputs/my_building.json
```

### Run a specific stage only

```bash
# Stage 1 only — NLP parsing
python building_nlp.py "2 floor house with 3 beds, 2 baths on a 30x40 plot"

# Stage 5 only — Structural grid from an existing layout file
python run_structural_grid.py --json outputs/floor_layout.json --floors 3

# Stage 8 only — Scheduling from an existing task list
python scheduler.py --tasks outputs/tasks.json
```

### Sample input file (`input.json`)

```json
{
  "requirement": "I want a 3 floor house with 4 bedrooms, 3 bathrooms, a kitchen, living room, dining room, study, and parking on a 40x60 plot.",
  "options": {
    "style": "modern",
    "n_floors": 3,
    "place_elevator": true
  }
}
```

```bash
python main_pipeline.py --file input.json --output outputs/result.json
```

### Output location

All outputs are saved to `ai_models/outputs/`:

```
outputs/
├── pipeline_result.json    ← Full JSON response
├── structural_grid.json    ← Column/beam/slab data
├── building_layout.json    ← Per-floor room coordinates
├── gantt.png               ← Gantt chart image
├── mep_routes.png          ← MEP routing visualization
├── cost_breakdown.png      ← Cost pie/bar chart
├── structural_floor_1.png  ← Floor 1 structural plan
├── structural_floor_2.png  ← Floor 2 structural plan
└── building_3d.html        ← Interactive 3D viewer
```

---

## 6. Backend Guide

### FastAPI application structure

```
backend/app/main.py        ← App factory, startup events, middleware
backend/app/routers/       ← One file per endpoint group
backend/app/jobs/tasks.py  ← Celery async tasks
backend/app/schemas/       ← Pydantic request/response models
```

### API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/generate` | Full pipeline — NLP to cost estimate |
| `POST` | `/generate/async` | Submit job, returns `job_id` immediately |
| `GET` | `/job/{job_id}` | Poll job status and retrieve result |
| `POST` | `/mep` | MEP routing for a given layout |
| `POST` | `/structural` | Structural grid for a given layout |
| `POST` | `/cost` | Cost estimate for a given building spec |
| `POST` | `/schedule` | CPM/PERT schedule for a task list |
| `GET` | `/health` | Health check — returns model readiness |
| `GET` | `/docs` | Auto-generated Swagger UI |
| `GET` | `/redoc` | ReDoc documentation |

### Example POST /generate request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "requirement": "I want a 2 floor house with 3 bedrooms, 2 bathrooms, kitchen, living room and parking on a 40x60 plot.",
    "save_outputs": true
  }'
```

### Example POST /generate/async (non-blocking)

```bash
# Step 1: Submit job
curl -X POST http://localhost:8000/generate/async \
  -H "Content-Type: application/json" \
  -d '{"requirement": "3 floor villa with 5 bedrooms on a 60x80 plot"}'

# Response:
# {"job_id": "a3f9b2c1-...", "status": "queued"}

# Step 2: Poll for result
curl http://localhost:8000/job/a3f9b2c1-...

# Response when complete:
# {"job_id": "...", "status": "complete", "result": {...}}
```

### Startup events

On application startup, FastAPI automatically:

1. Trains the column spacing GBR model (~2 seconds)
2. Trains the cost estimation GBR models (~3 seconds)
3. Loads RL agent weights if they exist in `ai_models/outputs/`
4. Connects to Redis and verifies the Celery queue
5. Logs readiness to stdout

---

## 7. Frontend Guide

### Technology stack

| Package | Purpose |
|---|---|
| React 18 | UI framework |
| Vite | Build tool and dev server |
| Axios | HTTP client for API calls |
| Recharts | Cost and schedule charts |
| Three.js | 3D building visualization |
| Tailwind CSS | Styling |

### How the frontend connects to the backend

The API base URL is configured via environment variable:

```bash
# frontend/.env
VITE_API_URL=http://localhost:8000
```

All API calls go through `frontend/src/api/client.js`, which sets this base URL automatically. In production, change this to your deployed backend URL.

### Development commands

```bash
cd frontend

# Install dependencies
npm install

# Start development server (with hot reload)
npm run dev

# Build for production
npm run build

# Preview production build locally
npm run preview

# Run linter
npm run lint
```

### Key components

**`InputForm.jsx`** — Text area for typing the building requirement. Calls `POST /generate/async` on submit and polls `GET /job/{id}` every 2 seconds until complete.

**`FloorPlanCanvas.jsx`** — Renders each floor's room layout as an SVG. Rooms are `<rect>` elements with colour coding by type. Supports zoom and floor switching.

**`StructuralView.jsx`** — Overlays columns (squares) and beams (lines) on the floor plan SVG. Colour-coded by reason (corner column vs. wet-stack column vs. grid column).

**`MEPView.jsx`** — Renders plumbing and electrical routes as coloured polylines on the floor plan. Toggle between plumbing and electrical layers.

**`GanttChart.jsx`** — Displays the CPM Gantt chart image returned by the backend. Alternatively renders an interactive timeline using the task JSON data.

**`CostDashboard.jsx`** — Pie chart of cost breakdown categories and bar chart comparing material, labor, and MEP totals. Powered by Recharts.

**`Viewer3D.jsx`** — Three.js canvas rendering the building in 3D. Rooms become extruded boxes with colour by type. Camera orbits on mouse drag.

---

## 8. Docker Setup

Docker is the recommended way to run the full system in production. It packages the backend, frontend, database, and job queue into a single command.

### Prerequisites

- Docker 24+ installed
- Docker Compose v2 installed

Verify:

```bash
docker --version
docker compose version
```

### 8.1 — Build all images

```bash
# From the project root
docker compose build
```

This builds:
- `bim-backend` — Python 3.11 + FastAPI + all AI models
- `bim-frontend` — Node 18 build, served by Nginx

### 8.2 — Start all services

```bash
# Development mode (with volume mounts for live editing)
docker compose up

# Production mode (detached, no volume mounts)
docker compose -f docker-compose.prod.yml up -d
```

### 8.3 — docker-compose.yml explained

```yaml
version: "3.9"

services:

  # ── FastAPI backend ─────────────────────────────────────
  backend:
    build:
      context: .
      dockerfile: docker/Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - MONGO_URL=${MONGO_URL}
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - mongodb
      - redis
    volumes:
      - ./ai_models:/app/ai_models    # mount AI modules
      - ./ai_models/outputs:/app/outputs  # persist model outputs
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000

  # ── Celery async worker ──────────────────────────────────
  worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.backend
    environment:
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=${DATABASE_URL}
    depends_on:
      - redis
    command: celery -A app.jobs.celery_app worker --loglevel=info --concurrency=4

  # ── React frontend (served by Nginx) ───────────────────
  frontend:
    build:
      context: .
      dockerfile: docker/Dockerfile.frontend
    ports:
      - "80:80"
    depends_on:
      - backend

  # ── PostgreSQL — stores project history ────────────────
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: bim_db
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # ── MongoDB — stores large JSON outputs ────────────────
  mongodb:
    image: mongo:7-jammy
    environment:
      MONGO_INITDB_DATABASE: bim_results
    volumes:
      - mongo_data:/data/db
    ports:
      - "27017:27017"

  # ── Redis — job queue for Celery ───────────────────────
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  mongo_data:
  redis_data:
```

### 8.4 — Useful Docker commands

```bash
# View running containers
docker compose ps

# View backend logs
docker compose logs -f backend

# View worker logs
docker compose logs -f worker

# Restart a single service
docker compose restart backend

# Stop everything
docker compose down

# Stop and delete all volumes (full reset)
docker compose down -v

# Run a one-off command inside the backend container
docker compose exec backend python ai_models/multifloor_train.py --episodes 300
```

---

## 9. Async Job Execution

Long-running operations (RL training, large buildings) are handled asynchronously using **Celery + Redis** so the API never times out.

### How it works

```
Client → POST /generate/async
              │
              ▼
      FastAPI enqueues task
      returns { job_id: "abc123" }
              │
              ▼
      Redis queue holds task
              │
              ▼
      Celery worker picks it up
      Runs full BIM pipeline
      Saves result to MongoDB
              │
              ▼
Client → GET /job/abc123
         ← { status: "complete", result: {...} }
```

### Job status values

| Status | Meaning |
|---|---|
| `queued` | Task submitted, waiting for a worker |
| `running` | Worker is processing the pipeline |
| `complete` | Result ready, fetch with `GET /job/{id}` |
| `failed` | Pipeline error — check `error` field in response |

### Starting the Celery worker

```bash
# Locally
celery -A app.jobs.celery_app worker --loglevel=info

# With Docker (already included in docker-compose.yml as the `worker` service)
docker compose up worker

# Monitor jobs with Flower (Celery dashboard)
pip install flower
celery -A app.jobs.celery_app flower --port=5555
# Open http://localhost:5555
```

### Checking queue depth

```bash
# Check how many tasks are waiting
redis-cli llen celery

# Flush queue (clear all pending jobs — use with caution)
redis-cli flushdb
```

---

## 10. Environment Variables

Copy `.env.example` to `.env` and fill in your values before starting any service.

```bash
cp .env.example .env
```

### `.env.example`

```dotenv
# ── Application ─────────────────────────────────────────
APP_ENV=development                   # development | production
APP_SECRET_KEY=change_this_to_a_random_string
LOG_LEVEL=info

# ── PostgreSQL ───────────────────────────────────────────
DATABASE_URL=postgresql://user:password@localhost:5432/bim_db
POSTGRES_USER=bim_user
POSTGRES_PASSWORD=bim_password
POSTGRES_DB=bim_db

# ── MongoDB ──────────────────────────────────────────────
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=bim_results

# ── Redis / Celery ───────────────────────────────────────
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# ── AI Model paths ───────────────────────────────────────
MODEL_OUTPUT_DIR=ai_models/outputs
RL_L1_AGENT_PATH=ai_models/outputs/l1_agent.pkl
RL_L2_AGENT_PATH=ai_models/outputs/l2_agent.pkl

# ── Frontend ─────────────────────────────────────────────
VITE_API_URL=http://localhost:8000

# ── Optional: OpenAI (for enhanced NLP) ─────────────────
# OPENAI_API_KEY=sk-...
```

> **Security:** Never commit your `.env` file. It is listed in `.gitignore` by default.

---

## 11. Testing

### Test the API is running

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "models_ready": true,
  "version": "1.0.0"
}
```

### Run the full pipeline via curl

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "requirement": "I want a 2 floor house with 3 bedrooms, 2 bathrooms, kitchen, living room and parking on a 40x60 plot."
  }' | python3 -m json.tool
```

### Test an async job

```bash
# 1. Submit
JOB=$(curl -s -X POST http://localhost:8000/generate/async \
  -H "Content-Type: application/json" \
  -d '{"requirement": "3 floor apartment block with 8 units"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

echo "Job ID: $JOB"

# 2. Poll until done
watch -n 2 "curl -s http://localhost:8000/job/$JOB | python3 -m json.tool"
```

### Run the Python unit tests

```bash
cd backend
pytest tests/ -v

# With coverage report
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

### Test individual AI modules

```bash
# Test NLP parsing
python -c "
from ai_models.building_nlp import BuildingNLPParser
p = BuildingNLPParser()
r = p.parse('2 floors, 3 beds, 2 baths, kitchen, 40x60 plot')
print(r.to_json())
"

# Test MEP routing
python -c "
from ai_models.mep_routing import MEPRouter
layout = [
  {'room':'bathroom','x':0,'y':0,'w':2,'h':2},
  {'room':'kitchen','x':4,'y':0,'w':4,'h':4},
]
router = MEPRouter(layout, columns=[[3.0,0.0],[3.0,3.0]])
result = router.route()
print(f'Plumbing routes: {len(result.plumbing_routes)}')
print(f'Electrical routes: {len(result.electrical_routes)}')
"
```

### Load test (requires `locust`)

```bash
pip install locust
locust -f tests/locustfile.py --host=http://localhost:8000
# Open http://localhost:8089 to run load test
```

---

## 12. Troubleshooting

### Module import errors

**Symptom:** `ModuleNotFoundError: No module named 'building_nlp'`

**Fix:** Make sure you are running commands from the correct directory and your virtual environment is active.

```bash
source venv/bin/activate
cd ai_models
python building_nlp.py "test input"
```

If running via Docker, ensure the `ai_models` directory is mounted as a volume (see `docker-compose.yml`).

---

**Symptom:** `ModuleNotFoundError: No module named 'networkx'` (or any other dependency)

**Fix:**

```bash
pip install -r backend/requirements.txt
```

If inside Docker:

```bash
docker compose build --no-cache backend
```

---

### Port conflicts

**Symptom:** `ERROR: address already in use — port 8000`

**Fix:** Find and kill the process using the port.

```bash
# Find process on port 8000
lsof -i :8000        # macOS / Linux
netstat -ano | findstr :8000   # Windows

# Kill it (replace PID with actual number)
kill -9 <PID>
```

Or change the port:

```bash
uvicorn app.main:app --port 8001
```

---

**Symptom:** `ERROR: address already in use — port 5173` (frontend)

**Fix:**

```bash
# Vite will auto-select next available port — check terminal for the actual URL
# Or set a custom port in vite.config.js:
# server: { port: 3000 }
```

---

### Docker issues

**Symptom:** `docker compose up` fails with `Cannot connect to Docker daemon`

**Fix:** Start Docker Desktop (macOS/Windows) or the Docker daemon (Linux):

```bash
sudo systemctl start docker
```

---

**Symptom:** Container builds succeed but backend crashes immediately

**Fix:** Check the logs:

```bash
docker compose logs backend
```

Common causes:
- Missing `.env` file — copy from `.env.example`
- Redis or Postgres not ready yet — add `healthcheck` to `docker-compose.yml` or increase `depends_on` retries

---

**Symptom:** Changes to Python files not reflected in Docker

**Fix:** Rebuild the image:

```bash
docker compose build backend
docker compose up backend
```

---

### Model loading errors

**Symptom:** `AssertionError: Call train() first` on startup

**Fix:** The structural grid and cost models train automatically on startup. This error means startup did not complete. Check for import errors earlier in the logs.

---

**Symptom:** RL agent produces poor floor plans (rooms overlapping or outside boundary)

**Fix:** The RL agents need more training episodes.

```bash
python ai_models/multifloor_train.py --episodes 500 --floors 3
```

More episodes = better layouts. Start with 300 for a working result, use 1000+ for production quality.

---

**Symptom:** `redis.exceptions.ConnectionError: Error connecting to Redis`

**Fix:** Ensure Redis is running:

```bash
# Check Redis status
redis-cli ping

# Start Redis
sudo systemctl start redis        # Linux
brew services start redis         # macOS
docker compose up redis           # Docker
```

---

**Symptom:** `celery.exceptions.NotRegistered` for pipeline tasks

**Fix:** Make sure the Celery worker imports the task module:

```bash
celery -A app.jobs.celery_app worker --loglevel=debug
```

Check that `app/jobs/tasks.py` is imported in `celery_app.py`.

---

### Slow pipeline execution

The pipeline normally completes in under 1 second for standard buildings. If it's slow:

- The RL agents are not loaded — they fall back to the slower rule-based planner. Train them with `multifloor_train.py`
- The structural grid ML models are re-training on every request. This is a configuration error — models should train once on startup, not per-request

---

## 13. Sample Output

The following is a representative JSON response from `POST /generate`.

```json
{
  "input_text": "I want a 3 floor house with 4 bedrooms, 3 bathrooms, kitchen, living room, dining room, study, and parking on a 40x60 plot.",

  "nlp": {
    "plot_width": 40.0,
    "plot_length": 60.0,
    "floors": 3,
    "rooms": {
      "bedroom": 4,
      "bathroom": 3,
      "kitchen": 1,
      "living_room": 1,
      "dining_room": 1,
      "study": 1
    },
    "parking": true,
    "garden": false
  },

  "building": {
    "floor_1": [
      {"room": "living_room",  "x": 0,  "y": 0, "w": 6, "h": 5, "floor": 1},
      {"room": "kitchen",      "x": 6,  "y": 0, "w": 4, "h": 4, "floor": 1},
      {"room": "dining_room",  "x": 6,  "y": 4, "w": 4, "h": 4, "floor": 1},
      {"room": "bathroom",     "x": 10, "y": 0, "w": 2, "h": 2, "floor": 1},
      {"room": "parking",      "x": 12, "y": 0, "w": 4, "h": 5, "floor": 1},
      {"room": "staircase",    "x": 0,  "y": 8, "w": 3, "h": 3, "floor": 1, "structural": true},
      {"room": "elevator",     "x": 3,  "y": 8, "w": 2, "h": 2, "floor": 1, "structural": true}
    ],
    "floor_2": [
      {"room": "master_bedroom","x": 0, "y": 0, "w": 5, "h": 4, "floor": 2},
      {"room": "bedroom",       "x": 5, "y": 0, "w": 4, "h": 4, "floor": 2},
      {"room": "bathroom",      "x": 9, "y": 0, "w": 2, "h": 2, "floor": 2},
      {"room": "study",         "x": 11,"y": 0, "w": 3, "h": 3, "floor": 2}
    ],
    "floor_3": [
      {"room": "bedroom",       "x": 0, "y": 0, "w": 4, "h": 4, "floor": 3},
      {"room": "bedroom",       "x": 4, "y": 0, "w": 4, "h": 4, "floor": 3},
      {"room": "bathroom",      "x": 8, "y": 0, "w": 2, "h": 2, "floor": 3}
    ]
  },

  "structural": {
    "floor_1": {
      "columns": [[0,0],[3,0],[6,0],[9,0],[12,0],[0,3],[3,3]],
      "beams": [
        {"start": [0,0], "end": [3,0], "span_m": 3.0, "direction": "X", "valid": true},
        {"start": [3,0], "end": [6,0], "span_m": 3.0, "direction": "X", "valid": true}
      ],
      "slabs": [
        {"corners": [[0,0],[3,0],[3,3],[0,3]], "area_m2": 9.0, "type": "two_way"}
      ],
      "validation": {
        "is_valid": true,
        "column_count": 31,
        "beam_count": 48,
        "slab_count": 36,
        "total_violations": 0
      }
    }
  },

  "mep": {
    "plumbing_routes": [
      [[10,1],[9,1],[8,1],[7,1],[6,1],[5,1],[4,1],[3,1],[2,1],[1,1],[0,1],[0,0]],
      [[6,2],[5,2],[4,2],[3,2],[2,2],[1,2],[0,2],[0,1],[0,0]]
    ],
    "electrical_routes": [
      [[0,19],[0,18],[0,17],[1,17],[2,17],[3,17],[3,12],[3,7],[3,2]],
      [[0,19],[1,19],[2,19],[3,19],[4,19],[4,14],[4,9],[4,4],[4,2]]
    ]
  },

  "tasks": [
    {"task": "site_clearing",      "depends": [],                      "duration_days": 2.0,  "crew": "civil"},
    {"task": "surveying",          "depends": ["site_clearing"],       "duration_days": 1.0,  "crew": "civil"},
    {"task": "excavation",         "depends": ["surveying"],           "duration_days": 7.1,  "crew": "civil"},
    {"task": "foundation",         "depends": ["excavation"],          "duration_days": 11.3, "crew": "structural"},
    {"task": "columns_f1",         "depends": ["waterproofing"],       "duration_days": 8.5,  "crew": "structural"},
    {"task": "plumbing_roughin",   "depends": ["internal_walls"],      "duration_days": 9.8,  "crew": "mep"},
    {"task": "electrical_roughin", "depends": ["internal_walls"],      "duration_days": 5.5,  "crew": "electrical"},
    {"task": "finishing",          "depends": ["painting"],            "duration_days": 6.0,  "crew": "finishing"},
    {"task": "handover",           "depends": ["final_inspection"],    "duration_days": 1.0,  "crew": "civil"}
  ],

  "schedule": {
    "project_duration_days": 604,
    "pert_p50_days": 618,
    "pert_p90_days": 674,
    "worker_peak": 42,
    "total_cost_usd": 3484160,
    "critical_path": [
      "site_clearing", "surveying", "excavation", "foundation",
      "columns_f1", "beams_f1", "slab_f1", "external_walls",
      "internal_walls", "plumbing_roughin", "flooring",
      "painting", "mep_testing", "final_inspection", "handover"
    ]
  },

  "cost": {
    "material_cost": 2562345,
    "labor_cost": 899026,
    "mep_cost": 1168389,
    "total_cost": 4329760,
    "contingency_10pct": 432976,
    "grand_total": 4762736,
    "cost_per_m2": 1984,
    "breakdown": {
      "foundation":    384351,
      "structure":     896820,
      "masonry":       384351,
      "finishes":      512468,
      "roofing":       205131,
      "mep_install":   701033,
      "fixtures":      467356,
      "civil_labor":   314659,
      "struct_labor":  314659,
      "finish_labor":  269708
    }
  },

  "timings_sec": {
    "1. NLP extraction":         0.01,
    "2. Constraint validation":  0.00,
    "3. Floor plan generation":  0.00,
    "4. Structural grid":        0.08,
    "5. MEP routing":            0.09,
    "6. Task generation":        0.00,
    "7. CPM/PERT scheduling":    0.00,
    "8. Cost estimation":        0.00,
    "total":                     0.19
  },

  "errors": []
}
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

Please run the test suite before opening a PR:

```bash
pytest backend/tests/ -v
npm run lint --prefix frontend
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ using FastAPI, React, NetworkX, scikit-learn, and NumPy.*
