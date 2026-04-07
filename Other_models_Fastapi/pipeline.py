"""
pipeline.py
===========
Full BIM Pipeline Orchestrator.

Flow:
  User Input (NLP text)
  → NLP Constraint Extractor   (building_nlp.py)
  → Constraint Validator        (constraint_validator.py)
  → RL Floor Plan Generator     (floor_plan_env + ppo_agent)
  → Multi-floor Planner         (multifloor_env)
  → Structural Grid             (structural_grid)
  → MEP Routing                 (mep_routing)
  → Construction Task Engine    (task_engine)
  → CPM/PERT Scheduler          (scheduler)
  → Cost Estimator              (cost_estimator)
  → Visualizations

All modules are imported and executed in sequence.
The full result is a single JSON dict (or FastAPI response).
"""

from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ── Module imports (all previous modules) ─────────────────────────────────
from building_nlp        import BuildingNLPParser
from constraint_validator import ConstraintValidator
from mep_routing          import MEPRouter
from task_engine          import ConstructionTaskEngine
from scheduler            import ProjectScheduler
from cost_estimator       import CostEstimationModel

# Structural grid (may not be in path — graceful fallback)
try:
    from structural_grid import MultiFloorStructuralGrid
    STRUCTURAL_OK = True
except ImportError:
    STRUCTURAL_OK = False

GRID = 20


# ─────────────────────────────────────────────────────────────────────────────
# Quick rule-based floor plan generator (no RL needed for pipeline demo)
# ─────────────────────────────────────────────────────────────────────────────

class QuickFloorPlanner:
    """
    Deterministic layout generator — packs rooms left-to-right, row-by-row.
    Used when the RL environment is not yet trained.
    """

    SIZES = {
        "living_room":    (6,5), "kitchen":    (4,4), "dining_room": (4,4),
        "master_bedroom": (5,4), "bedroom":    (4,4), "bathroom":    (2,2),
        "study":          (3,3), "parking":    (4,5), "home_office": (3,3),
        "gym":            (4,4), "staircase":  (3,3), "elevator":    (2,2),
        "storage":        (2,2), "laundry":    (2,2), "terrace":     (3,3),
        "balcony":        (2,2), "dining":     (3,3), "guest_room":  (3,4),
    }
    DEFAULT_SIZE = (3, 3)

    GROUND_ROOMS  = ["living_room","kitchen","dining_room","bathroom","parking"]
    UPPER_ROOMS   = ["master_bedroom","bedroom","bedroom","bathroom","study"]

    def plan(self, rooms_spec: dict, n_floors: int) -> dict[str, list[dict]]:
        building = {}
        for fn in range(1, n_floors + 1):
            if fn == 1:
                room_types = self._ground_from_spec(rooms_spec)
            else:
                room_types = self._upper_from_spec(rooms_spec)
            building[f"floor_{fn}"] = self._pack(room_types, fn)
        return building

    def _ground_from_spec(self, spec: dict) -> list[str]:
        rooms = []
        r = spec.get("rooms", {})
        if r.get("living_room",0) > 0:  rooms.append("living_room")
        if r.get("kitchen",0)    > 0:  rooms.append("kitchen")
        if r.get("dining_room",0)> 0:  rooms.append("dining_room")
        if r.get("bathroom",0)   > 0:  rooms.append("bathroom")
        if spec.get("parking"):        rooms.append("parking")
        if not rooms:
            rooms = list(self.GROUND_ROOMS)
        return rooms

    def _upper_from_spec(self, spec: dict) -> list[str]:
        rooms = []
        r = spec.get("rooms", {})
        beds = r.get("bedroom", 0) + r.get("master_bedroom", 0)
        if beds > 0:
            rooms.append("master_bedroom")
            for _ in range(max(0, beds - 1)):
                rooms.append("bedroom")
        baths = r.get("bathroom", 0)
        for _ in range(max(1, baths)):
            rooms.append("bathroom")
        if r.get("study", 0) > 0:     rooms.append("study")
        if r.get("home_office",0) > 0: rooms.append("home_office")
        if not rooms:
            rooms = list(self.UPPER_ROOMS)
        return rooms

    def _pack(self, room_types: list[str], floor_num: int) -> list[dict]:
        placed = []
        x, y, row_h = 0, 0, 0

        # Add staircase first (upper floors)
        if floor_num > 1:
            placed.append({"room":"staircase","x":0,"y":GRID-4,"w":3,"h":3,
                           "floor":floor_num,"structural":True})
            placed.append({"room":"elevator", "x":3,"y":GRID-4,"w":2,"h":2,
                           "floor":floor_num,"structural":True})

        for rtype in room_types:
            w, h = self.SIZES.get(rtype, self.DEFAULT_SIZE)
            if x + w > GRID:
                y += row_h; x = 0; row_h = 0
            if y + h > GRID - 4:
                break
            placed.append({"room":rtype,"x":x,"y":y,"w":w,"h":h,"floor":floor_num})
            x += w
            row_h = max(row_h, h)

        return placed


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    input_text:    str
    nlp:           dict
    constraints:   dict
    building:      dict   # {"floor_1": [...], ...}
    structural:    dict   # {"floor_1": {"columns":[],"beams":[],"slabs":[]}}
    mep:           dict   # {"plumbing_routes":[], "electrical_routes":[]}
    tasks:         list[dict]
    schedule:      dict
    cost:          dict
    timings:       dict
    errors:        list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "input_text":  self.input_text,
            "nlp":         self.nlp,
            "constraints": self.constraints,
            "building":    self.building,
            "structural":  self.structural,
            "mep":         self.mep,
            "tasks":       self.tasks,
            "schedule":    self.schedule,
            "cost":        self.cost,
            "timings_sec": self.timings,
            "errors":      self.errors,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        c = self.cost
        s = self.schedule
        lines = [
            "╔" + "═"*56 + "╗",
            "║  BIM PIPELINE — FULL BUILD SUMMARY                    ║",
            "╠" + "═"*56 + "╣",
            f"║  Input       : {self.input_text[:50]:<50} ║",
            f"║  Floors      : {self.nlp.get('floors',1):<50} ║",
            f"║  Rooms       : {len([r for fl in self.building.values() for r in fl if not r.get('structural')]):<50} ║",
            f"║  Columns     : {sum(len(v.get('columns',[])) for v in self.structural.values()):<50} ║",
            f"║  Plumb runs  : {len(self.mep.get('plumbing_routes',[])):<50} ║",
            f"║  Elec runs   : {len(self.mep.get('electrical_routes',[])):<50} ║",
            f"║  Tasks       : {len(self.tasks):<50} ║",
            f"║  Duration    : {s.get('project_duration_days',0):.0f} days (P90: {s.get('pert_p90_days',0):.0f}d){'':<24} ║",
            f"║  Material    : ${c.get('material_cost',0):>12,.0f}{'':<36} ║",
            f"║  Labor       : ${c.get('labor_cost',0):>12,.0f}{'':<36} ║",
            f"║  Grand Total : ${c.get('grand_total',0):>12,.0f}{'':<36} ║",
            f"║  Cost/m²     : ${c.get('cost_per_m2',0):>12,.0f}{'':<36} ║",
            "╚" + "═"*56 + "╝",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class BIMPipeline:
    """
    Full Building Information Model generation pipeline.
    Call run(text) to execute all stages.
    """

    def __init__(self, verbose: bool = True):
        self.verbose    = verbose
        self._nlp       = BuildingNLPParser()
        self._cv        = ConstraintValidator()
        self._planner   = QuickFloorPlanner()
        self._sched     = ProjectScheduler()
        self._cost      = CostEstimationModel()
        self._struct    = MultiFloorStructuralGrid() if STRUCTURAL_OK else None
        self._models_ready = False

    def prepare(self):
        """Pre-train all ML models. Call once before run()."""
        if self.verbose:
            print("\n── Preparing ML models ─────────────────────────────────")
        if self._struct:
            self._struct.train(verbose=self.verbose)
        self._cost.train(verbose=self.verbose)
        self._models_ready = True
        if self.verbose:
            print("── Models ready ────────────────────────────────────────\n")

    def run(self, text: str) -> PipelineResult:
        t_total = time.time()
        timings = {}
        errors  = []

        def step(name: str, fn):
            t0 = time.time()
            try:
                result = fn()
                timings[name] = round(time.time() - t0, 3)
                if self.verbose:
                    print(f"  ✓ {name:<28} {timings[name]:.2f}s")
                return result
            except Exception as e:
                timings[name]  = round(time.time() - t0, 3)
                errors.append(f"{name}: {str(e)}")
                if self.verbose:
                    print(f"  ✗ {name:<28} ERROR: {e}")
                return None

        if self.verbose:
            print("\n" + "═"*60)
            print("  BIM PIPELINE EXECUTING")
            print("═"*60)

        # ── Stage 1: NLP ─────────────────────────────────────────────
        nlp_result = step("1. NLP extraction",
            lambda: self._nlp.parse(text))
        nlp_dict = nlp_result.to_dict() if nlp_result else {"floors":1,"rooms":{"bedroom":2},"parking":False}
        nlp_dict.pop("warnings", None)

        # ── Stage 2: Constraint validation ───────────────────────────
        cv_result = step("2. Constraint validation",
            lambda: self._cv.validate(nlp_dict))
        cv_dict = cv_result.to_dict() if cv_result else {}

        # ── Stage 3: Floor plan generation ───────────────────────────
        n_floors  = nlp_dict.get("floors", 1)
        building  = step("3. Floor plan generation",
            lambda: self._planner.plan(nlp_dict, n_floors))
        if building is None:
            building = {"floor_1":[{"room":"living_room","x":0,"y":0,"w":5,"h":5}]}

        # ── Stage 4: Structural grid ──────────────────────────────────
        if self._struct and self._models_ready:
            structural_all = step("4. Structural grid",
                lambda: self._struct.generate_all(building, verbose=False))
            structural_dict = {
                f"floor_{fn}": g.to_dict()
                for fn, g in (structural_all or {}).items()
            }
        else:
            structural_dict = self._fallback_structural(building)
            timings["4. Structural grid"] = 0.0

        # Combined columns for MEP
        all_cols = []
        for fd in structural_dict.values():
            all_cols.extend(fd.get("columns", []))

        # ── Stage 5: MEP routing ──────────────────────────────────────
        all_rooms = [r for fl in building.values() for r in fl]
        core_dict = {}

        def do_mep():
            router = MEPRouter(all_rooms, all_cols, core_dict)
            return router.route()

        mep_result = step("5. MEP routing", do_mep)
        mep_dict   = mep_result.path_coords() if mep_result else {"plumbing_routes":[],"electrical_routes":[]}

        # ── Stage 6: Construction tasks ───────────────────────────────
        area_m2    = (nlp_dict.get("plot_width") or 20) * (nlp_dict.get("plot_length") or 20)
        s1_struct  = list(structural_dict.values())[0] if structural_dict else {}

        def do_tasks():
            eng = ConstructionTaskEngine()
            eng.generate(
                rooms      = all_rooms,
                structural = s1_struct,
                mep        = mep_dict,
                n_floors   = n_floors,
                area_m2    = float(area_m2),
            )
            return eng

        task_eng = step("6. Task generation", do_tasks)
        tasks_list = [t.to_dict() for t in (task_eng.tasks if task_eng else [])]

        # ── Stage 7: CPM/PERT scheduling ─────────────────────────────
        def do_schedule():
            return self._sched.compute(task_eng.tasks, task_eng.get_dag())

        schedule_result = step("7. CPM/PERT scheduling", do_schedule)
        schedule_dict   = schedule_result.to_dict() if schedule_result else {}

        # ── Stage 8: Cost estimation ──────────────────────────────────
        def do_cost():
            return self._cost.estimate(
                area_m2    = float(area_m2),
                n_rooms    = len(all_rooms),
                n_floors   = n_floors,
                rooms      = all_rooms,
                structural = s1_struct,
                mep        = mep_dict,
            )

        cost_result = step("8. Cost estimation", do_cost)
        cost_dict   = cost_result.to_dict() if cost_result else {}

        timings["total"] = round(time.time() - t_total, 3)
        if self.verbose:
            print(f"\n  Total pipeline time: {timings['total']}s")
            if errors:
                print(f"  Errors: {len(errors)}")

        return PipelineResult(
            input_text   = text,
            nlp          = nlp_dict,
            constraints  = cv_dict,
            building     = building,
            structural   = structural_dict,
            mep          = mep_dict,
            tasks        = tasks_list,
            schedule     = schedule_dict,
            cost         = cost_dict,
            timings      = timings,
            errors       = errors,
        )

    def _fallback_structural(self, building: dict) -> dict:
        """Simple column grid fallback when structural module unavailable."""
        result = {}
        for key, layout in building.items():
            cols, beams = [], []
            for x in range(0, GRID+1, 3):
                for y in range(0, GRID+1, 3):
                    cols.append([float(x), float(y)])
            result[key] = {"columns": cols, "beams": beams, "slabs": []}
        return result


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI-compatible app definition
# ─────────────────────────────────────────────────────────────────────────────

class BIMApp:
    """
    FastAPI-compatible application.
    In production: wrap with @app.post("/generate") decorators.
    Standalone: call app.generate(text) directly.
    """

    def __init__(self):
        self.pipeline = BIMPipeline(verbose=True)
        self._ready   = False

    def startup(self):
        """Call on app startup — trains all ML models."""
        self.pipeline.prepare()
        self._ready = True

    def generate(self, requirement_text: str, save_outputs: bool = True) -> dict:
        """
        POST /generate  — Main API endpoint.
        Input : {"requirement": "I want a 2 floor house..."}
        Output: Full BIM JSON
        """
        if not self._ready:
            self.startup()

        result = self.pipeline.run(requirement_text)

        if save_outputs:
            out = Path("outputs"); out.mkdir(exist_ok=True)
            with open(out / "pipeline_result.json", "w") as f:
                json.dump(result.to_dict(), f, indent=2)

            # Gantt chart
            if result.schedule.get("tasks"):
                from scheduler import Schedule, ScheduledTask
                try:
                    sched_obj = self._rebuild_schedule(result.schedule)
                    self.pipeline._sched.gantt_chart(
                        sched_obj, save_path=str(out/"gantt.png")
                    )
                except Exception as e:
                    print(f"  Gantt chart error: {e}")

        print("\n" + result.summary())
        return result.to_dict()

    def _rebuild_schedule(self, schedule_dict: dict):
        """Rebuild Schedule object from dict for visualization."""
        from scheduler import Schedule, ScheduledTask
        tasks = []
        for td in schedule_dict.get("tasks", []):
            st = ScheduledTask(
                task=td["task"], crew=td["crew"],
                duration=td["duration_days"], workers=td["workers"],
                cost_usd=td["cost_usd"], description=td.get("description",""),
                es=td["es"], ef=td["ef"], ls=td["ls"], lf=td["lf"],
                tf=td["float_days"],
                pert_expected=td.get("pert_expected",td["duration_days"]),
                pert_variance=td.get("pert_std",0)**2,
                on_critical_path=td["on_critical_path"],
            )
            tasks.append(st)
        return Schedule(
            tasks=tasks,
            project_duration=schedule_dict["project_duration_days"],
            critical_path=schedule_dict["critical_path"],
            total_cost=schedule_dict["total_cost_usd"],
            pert_p50=schedule_dict.get("pert_p50_days",0),
            pert_p90=schedule_dict.get("pert_p90_days",0),
            worker_peak=schedule_dict.get("worker_peak",0),
        )
