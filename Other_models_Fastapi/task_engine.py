"""
task_engine.py
==============
Construction Task Generation Engine.

Generates a full construction task DAG from the complete building model.
Tasks are ordered by engineering dependencies (can't wire before walls exist).

Output format:
  [{"task": "foundation", "depends": [], "duration_days": 10, "crew": "civil"}, ...]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx


# ─────────────────────────────────────────────────────────────────────────────
# Task catalogue
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskDef:
    name:          str
    crew_type:     str        # "civil" | "structural" | "mep" | "electrical" | "finishing"
    base_days:     float      # base duration (scaled by area/count)
    depends_on:    list[str]  = field(default_factory=list)
    workers:       int        = 4
    cost_per_day:  float      = 500.0   # USD/day base
    description:   str        = ""
    floor_specific:bool       = False   # replicated per floor


# Master task catalogue (construction sequencing logic)
TASK_CATALOGUE: list[TaskDef] = [
    # ── Site & Foundation ──────────────────────────────────────────────
    TaskDef("site_clearing",      "civil",       2,  [],                              2,  300,  "Clear & level site"),
    TaskDef("surveying",          "civil",       1,  ["site_clearing"],               2,  400,  "Mark column grid"),
    TaskDef("excavation",         "civil",       5,  ["surveying"],                   6,  600,  "Excavate to founding depth"),
    TaskDef("foundation",         "structural",  8,  ["excavation"],                  8,  900,  "Pour RC strip/raft foundation"),
    TaskDef("waterproofing",      "mep",         3,  ["foundation"],                  4,  500,  "Foundation waterproofing"),

    # ── Structural frame ───────────────────────────────────────────────
    TaskDef("columns_f1",         "structural",  6,  ["waterproofing"],               6,  800,  "RC columns ground floor",  True),
    TaskDef("beams_f1",           "structural",  4,  ["columns_f1"],                  6,  800,  "RC beams ground floor",    True),
    TaskDef("slab_f1",            "structural",  5,  ["beams_f1"],                    8,  700,  "RC slab ground floor",     True),
    TaskDef("columns_upper",      "structural",  5,  ["slab_f1"],                     6,  800,  "RC columns upper floors",  True),
    TaskDef("beams_upper",        "structural",  4,  ["columns_upper"],               6,  800,  "RC beams upper floors",    True),
    TaskDef("slab_upper",         "structural",  5,  ["beams_upper"],                 8,  700,  "RC slab upper floors",     True),
    TaskDef("staircase",          "structural",  4,  ["slab_f1"],                     4,  700,  "Staircase construction"),
    TaskDef("elevator_shaft",     "structural",  6,  ["columns_f1"],                  4,  900,  "Elevator shaft walls"),
    TaskDef("roof_structure",     "structural",  5,  ["slab_upper"],                  6,  750,  "Roof slab/structure"),

    # ── Masonry & Walls ────────────────────────────────────────────────
    TaskDef("external_walls",     "civil",       8,  ["slab_f1"],                     8,  600,  "External brickwork/blockwork"),
    TaskDef("internal_walls",     "civil",       7,  ["external_walls"],              8,  550,  "Internal partition walls"),
    TaskDef("wall_plastering",    "finishing",   6,  ["internal_walls"],              6,  450,  "Plaster internal/external walls"),

    # ── MEP rough-in ──────────────────────────────────────────────────
    TaskDef("plumbing_roughin",   "mep",         6,  ["internal_walls"],              4,  650,  "Plumbing pipes & waste runs"),
    TaskDef("electrical_roughin", "electrical",  5,  ["internal_walls"],              4,  600,  "Conduit & cable trays"),
    TaskDef("hvac_roughin",       "mep",         4,  ["internal_walls"],              4,  700,  "HVAC ductwork & grilles"),
    TaskDef("plumbing_stack",     "mep",         3,  ["plumbing_roughin"],            3,  600,  "Vertical plumbing stacks"),
    TaskDef("electrical_panel",   "electrical",  2,  ["electrical_roughin"],          2,  700,  "Main distribution boards"),
    TaskDef("sewer_connection",   "mep",         2,  ["plumbing_roughin"],            3,  500,  "Connect to municipal sewer"),

    # ── Windows & Doors ────────────────────────────────────────────────
    TaskDef("windows",            "finishing",   4,  ["external_walls"],              4,  500,  "Window frame & glazing installation"),
    TaskDef("doors_external",     "finishing",   2,  ["external_walls"],              2,  500,  "External doors"),
    TaskDef("doors_internal",     "finishing",   3,  ["internal_walls"],              3,  450,  "Internal door frames & shutters"),

    # ── Finishing ─────────────────────────────────────────────────────
    TaskDef("flooring",           "finishing",   6,  ["wall_plastering", "plumbing_roughin"],  8,  600,  "Floor screed + tiling"),
    TaskDef("ceiling",            "finishing",   4,  ["electrical_roughin", "hvac_roughin"],   6,  500,  "False ceiling & cornices"),
    TaskDef("painting",           "finishing",   5,  ["wall_plastering", "ceiling"],   6,  400,  "Interior & exterior painting"),
    TaskDef("kitchen_fit",        "finishing",   3,  ["plumbing_roughin","electrical_roughin"], 3, 600, "Kitchen cabinet & countertop"),
    TaskDef("bathroom_fit",       "finishing",   3,  ["plumbing_roughin","flooring"],  3,  600,  "Bathroom fixtures & tiling"),
    TaskDef("electrical_finish",  "electrical",  4,  ["painting"],                    4,  550,  "Sockets, switches, lighting"),
    TaskDef("plumbing_finish",    "mep",         3,  ["bathroom_fit","kitchen_fit"],   3,  600,  "Sanitary fixtures & taps"),
    TaskDef("hvac_finish",        "mep",         3,  ["ceiling"],                     3,  650,  "AC units & grille covers"),

    # ── External works ────────────────────────────────────────────────
    TaskDef("external_finishes",  "finishing",   4,  ["painting"],                    4,  500,  "Exterior cladding & rendering"),
    TaskDef("landscaping",        "civil",       3,  ["external_finishes"],           4,  400,  "Garden, driveway, fencing"),
    TaskDef("elevator_install",   "mep",         5,  ["elevator_shaft","slab_upper"], 2, 1200,  "Elevator car & machinery"),

    # ── Inspections & Handover ────────────────────────────────────────
    TaskDef("mep_testing",        "mep",         2,  ["plumbing_finish","electrical_finish","hvac_finish"], 3, 500, "MEP pressure tests & commissioning"),
    TaskDef("structural_inspect", "structural",  1,  ["roof_structure"],              2,  600,  "Structural certification"),
    TaskDef("final_inspection",   "civil",       2,  ["mep_testing","structural_inspect","landscaping"],  4, 500, "Building completion certificate"),
    TaskDef("handover",           "civil",       1,  ["final_inspection"],            2,  300,  "Keys & documentation handover"),
]

TASK_BY_NAME = {t.name: t for t in TASK_CATALOGUE}


# ─────────────────────────────────────────────────────────────────────────────
# Task instance
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    task:        str
    crew:        str
    depends:     list[str]
    duration:    float
    workers:     int
    cost_usd:    float
    description: str
    floor:       Optional[int] = None

    def to_dict(self) -> dict:
        d = {
            "task":        self.task,
            "depends":     self.depends,
            "duration_days": round(self.duration, 1),
            "crew":        self.crew,
            "workers":     self.workers,
            "cost_usd":    round(self.cost_usd),
            "description": self.description,
        }
        if self.floor is not None:
            d["floor"] = self.floor
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Task Engine
# ─────────────────────────────────────────────────────────────────────────────

class ConstructionTaskEngine:
    """
    Generates a construction task DAG from the full building model.

    Scaling rules:
      - Duration scaled by √(area/100) — bigger buildings take longer
      - Workers scaled by floor count
      - MEP tasks scaled by route count
    """

    def __init__(self):
        self.G: nx.DiGraph = nx.DiGraph()
        self.tasks: list[Task] = []

    def generate(
        self,
        rooms:       list[dict],
        structural:  dict,
        mep:         Optional[dict]  = None,
        n_floors:    int = 1,
        area_m2:     float = 200.0,
    ) -> list[Task]:

        area_factor = max(math.sqrt(area_m2 / 100), 0.5)
        floor_factor= max(math.sqrt(n_floors), 1.0)

        n_bathrooms = sum(1 for r in rooms if "bathroom" in r.get("room",""))
        n_kitchens  = sum(1 for r in rooms if r.get("room","") == "kitchen")
        n_rooms     = len(rooms)
        has_elevator= any(r.get("room","") == "elevator" for r in rooms)
        has_parking = any(r.get("room","") == "parking" for r in rooms)

        mep_factor  = max(1.0, (n_bathrooms + n_kitchens) * 0.3)
        plumb_routes= len(mep.get("plumbing_routes", [])) if mep else n_bathrooms
        elec_routes = len(mep.get("electrical_routes",[])) if mep else n_rooms

        tasks_out: list[Task] = []

        for tdef in TASK_CATALOGUE:
            # Skip elevator if not in building
            if tdef.name == "elevator_install" and not has_elevator:
                continue
            if tdef.name == "elevator_shaft" and not has_elevator:
                continue

            # Scale duration
            dur = tdef.base_days * area_factor
            if "upper" in tdef.name:
                dur *= floor_factor
            if tdef.crew_type == "mep":
                dur *= mep_factor
            if tdef.name in ("plumbing_roughin","plumbing_stack"):
                dur *= max(1.0, plumb_routes * 0.4)
            if tdef.name in ("electrical_roughin",):
                dur *= max(1.0, elec_routes  * 0.15)

            dur = round(max(dur, 1.0), 1)

            # Scale workers
            workers = tdef.workers
            if n_floors > 2:
                workers = min(workers + 2, 16)

            cost    = dur * workers * tdef.cost_per_day

            # Resolve dependencies (filter only existing tasks)
            all_names = {td.name for td in TASK_CATALOGUE
                         if td.name != "elevator_install" or has_elevator}

            depends = [d for d in tdef.depends_on if d in all_names]

            task = Task(
                task        = tdef.name,
                crew        = tdef.crew_type,
                depends     = depends,
                duration    = dur,
                workers     = workers,
                cost_usd    = cost,
                description = tdef.description,
            )
            tasks_out.append(task)

        # Build DAG
        self.G.clear()
        for t in tasks_out:
            self.G.add_node(t.task)
        for t in tasks_out:
            for dep in t.depends:
                if self.G.has_node(dep):
                    self.G.add_edge(dep, t.task)

        # Validate DAG
        if not nx.is_directed_acyclic_graph(self.G):
            cycles = list(nx.find_cycle(self.G))
            raise ValueError(f"Task DAG has cycles: {cycles}")

        # Topological sort
        topo_order = list(nx.topological_sort(self.G))
        task_map   = {t.task: t for t in tasks_out}
        self.tasks = [task_map[n] for n in topo_order if n in task_map]

        return self.tasks

    def get_dag(self) -> nx.DiGraph:
        return self.G

    def critical_tasks(self) -> list[str]:
        """Tasks with zero float (on the critical path)."""
        # Computed by CPM in scheduler — placeholder here
        return [t.task for t in self.tasks if t.crew == "structural"]


import math   # needed inside class methods
