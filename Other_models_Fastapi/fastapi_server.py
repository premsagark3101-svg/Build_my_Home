"""
fastapi_server.py
=================
FastAPI backend exposing the full BIM pipeline as REST endpoints.

Endpoints:
  POST /generate          — Full pipeline from NLP text
  POST /mep               — MEP routing only
  POST /schedule          — Scheduling only
  POST /cost              — Cost estimation only
  GET  /health            — Health check

Run with:
  pip install fastapi uvicorn
  uvicorn fastapi_server:app --reload --port 8000

Without FastAPI (offline):
  python fastapi_server.py    ← runs the demo pipeline directly
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Try FastAPI, fall back to demo mode ───────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from pipeline import BIMApp

# ─────────────────────────────────────────────────────────────────────────────
# Request/Response models (Pydantic or plain dict)
# ─────────────────────────────────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    class GenerateRequest(BaseModel):
        requirement: str
        save_outputs: bool = True

    class MEPRequest(BaseModel):
        layout:   list[dict]
        columns:  list[list[float]]
        core:     dict = {}

    class CostRequest(BaseModel):
        area_m2:   float
        n_rooms:   int
        n_floors:  int
        rooms:     list[dict]
        structural:dict = {}

# ─────────────────────────────────────────────────────────────────────────────
# App instance
# ─────────────────────────────────────────────────────────────────────────────

bim_app = BIMApp()


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title       = "BIM Pipeline API",
        description = "Full Building Information Model generation from NLP requirements",
        version     = "1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup():
        bim_app.startup()

    @app.get("/health")
    async def health():
        return {"status": "ok", "models_ready": bim_app._ready}

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        """
        Full pipeline: NLP → Floor Plan → Structure → MEP → Schedule → Cost.
        """
        try:
            result = bim_app.generate(req.requirement, req.save_outputs)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/mep")
    async def mep_only(req: MEPRequest):
        """Route MEP for a given layout and column set."""
        from mep_routing import MEPRouter
        router = MEPRouter(req.layout, req.columns, req.core)
        result = router.route()
        return result.path_coords()

    @app.post("/cost")
    async def cost_only(req: CostRequest):
        """Estimate cost for a given building spec."""
        from cost_estimator import CostEstimationModel
        m = CostEstimationModel()
        m.train(verbose=False)
        est = m.estimate(
            area_m2=req.area_m2, n_rooms=req.n_rooms,
            n_floors=req.n_floors, rooms=req.rooms,
            structural=req.structural,
        )
        return est.to_dict()

    @app.get("/pipeline/schema")
    async def schema():
        """Return the pipeline I/O schema."""
        return {
            "input":  {"requirement": "string — natural language building description"},
            "output": {
                "nlp":          "extracted constraints",
                "building":     "per-floor room layouts",
                "structural":   "columns, beams, slabs per floor",
                "mep":          "plumbing_routes, electrical_routes",
                "tasks":        "construction task DAG",
                "schedule":     "CPM/PERT timeline",
                "cost":         "material, labor, total cost",
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# Visualization for pipeline output
# ─────────────────────────────────────────────────────────────────────────────

def visualize_mep(mep: dict, building: dict, save_path: str = None):
    """Visualize MEP routes over floor plan."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import to_rgba

    GRID = 20
    ROOM_COLORS = {
        "living_room":"#2E8B57","kitchen":"#FF8C00","dining_room":"#6B8E23",
        "bedroom":"#4169E1","master_bedroom":"#1E3A8A","bathroom":"#20B2AA",
        "study":"#9932CC","parking":"#708090","staircase":"#DC143C",
        "elevator":"#B8860B",
    }

    layout_f1 = building.get("floor_1", [])
    plumb  = mep.get("plumbing_routes",   [])
    elec   = mep.get("electrical_routes", [])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("#0a0f1e")

    for ax, routes, title, color in [
        (axes[0], plumb, "Plumbing Routes (A*)",    "#00FFFF"),
        (axes[1], elec,  "Electrical Routes (Dijkstra)", "#FFD700"),
    ]:
        ax.set_facecolor("#0d1b2a")
        for i in range(GRID+1):
            ax.axhline(i, color="#ffffff08", lw=0.3)
            ax.axvline(i, color="#ffffff08", lw=0.3)

        # Rooms
        for r in layout_f1:
            c = ROOM_COLORS.get(r["room"],"#555")
            ax.add_patch(patches.FancyBboxPatch(
                (r["x"]+0.05, r["y"]+0.05), r["w"]-0.1, r["h"]-0.1,
                boxstyle="round,pad=0.05",
                facecolor=to_rgba(c,0.4), edgecolor=to_rgba(c,0.7), lw=0.8, zorder=2,
            ))
            ax.text(r["x"]+r["w"]/2, r["y"]+r["h"]/2,
                    r["room"].replace("_","\n"), ha="center", va="center",
                    fontsize=5.5, color="white", alpha=0.7)

        # Routes
        cmap = plt.cm.cool if color=="#00FFFF" else plt.cm.autumn
        for i, path in enumerate(routes):
            if not path: continue
            xs = [p[0]+0.5 for p in path]
            ys = [p[1]+0.5 for p in path]
            c_ = cmap(i / max(len(routes),1))
            ax.plot(xs, ys, color=c_, lw=1.8, alpha=0.8, zorder=5)
            ax.plot(xs[0],  ys[0],  "o", color="lime",    ms=5, zorder=6)
            ax.plot(xs[-1], ys[-1], "s", color="#FF4444", ms=5, zorder=6)

        ax.add_patch(patches.Rectangle((0,0),GRID,GRID,
                     lw=2, edgecolor="#FF6B6B", facecolor="none", zorder=10))
        ax.set_xlim(0,GRID); ax.set_ylim(0,GRID); ax.set_aspect("equal")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white", labelsize=7)

    fig.suptitle("MEP Routing — NetworkX A* + Dijkstra",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  MEP visualization → {save_path}")
    return fig


def visualize_cost(cost: dict, save_path: str = None):
    """Pie/bar breakdown of cost estimate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("#0a0f1e")

    brkdwn = cost.get("breakdown", {})
    labels = [k.replace("_"," ").title() for k in brkdwn]
    values = list(brkdwn.values())
    colors = plt.cm.Set3(np.linspace(0,1,len(values)))

    ax0 = axes[0]
    ax0.set_facecolor("#0d1b2a")
    wedges, texts, autotexts = ax0.pie(
        values, labels=labels, colors=colors,
        autopct="%1.0f%%", startangle=140,
        textprops={"color":"white","fontsize":7},
        pctdistance=0.75,
    )
    for at in autotexts: at.set_fontsize(6)
    ax0.set_title("Cost Breakdown", color="white", fontsize=11, fontweight="bold")

    ax1 = axes[1]
    ax1.set_facecolor("#0d1b2a")
    cats   = ["Material","Labor","MEP","Contingency"]
    vals   = [cost.get("material_cost",0), cost.get("labor_cost",0),
              cost.get("mep_cost",0),      cost.get("contingency_10pct",0)]
    cols   = ["#4682B4","#3CB371","#20B2AA","#FF6B6B"]
    bars   = ax1.barh(cats, vals, color=cols, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_width()+1000, bar.get_y()+bar.get_height()/2,
                 f"${val:,.0f}", va="center", color="white", fontsize=9)
    ax1.set_xlabel("USD", color="white")
    ax1.set_title(f"Total: ${cost.get('grand_total',0):,.0f}  |  ${cost.get('cost_per_m2',0):,.0f}/m²",
                  color="white", fontsize=10, fontweight="bold")
    ax1.tick_params(colors="white")
    for sp in ax1.spines.values(): sp.set_edgecolor("#333")

    fig.suptitle("Construction Cost Estimate — ML Model",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Cost visualization → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Demo / CLI runner
# ─────────────────────────────────────────────────────────────────────────────

DEMO_INPUT = (
    "I want a 3 floor house with 4 bedrooms, 3 bathrooms, kitchen, "
    "living room, dining room, study, parking and a garden on a 40x60 plot."
)

if __name__ == "__main__":
    if FASTAPI_AVAILABLE and "--serve" in sys.argv:
        import uvicorn
        print("Starting BIM API server on http://localhost:8000")
        uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000, reload=False)
    else:
        # Standalone demo
        out = Path("outputs"); out.mkdir(exist_ok=True)

        bim_app.startup()
        result_dict = bim_app.generate(DEMO_INPUT, save_outputs=True)

        mep = result_dict.get("mep", {})
        building = result_dict.get("building", {})
        cost = result_dict.get("cost", {})

        visualize_mep(mep, building, save_path=str(out/"mep_routes.png"))
        visualize_cost(cost, save_path=str(out/"cost_breakdown.png"))

        print(f"\nAll outputs saved to: {out.absolute()}")
        print("\nTo run as API server:")
        print("  pip install fastapi uvicorn")
        print("  python fastapi_server.py --serve")
        print("\nAPI endpoints:")
        print("  POST /generate  — full pipeline")
        print("  POST /mep       — MEP routing only")
        print("  POST /cost      — cost estimation only")
        print("  GET  /health    — health check")
