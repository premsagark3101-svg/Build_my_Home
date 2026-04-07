"""
scheduler.py
============
Project Scheduling Module using Critical Path Method (CPM) and PERT.

Inputs:  Task list with dependencies and durations
Outputs:
  • Early/late start & finish (CPM forward/backward pass)
  • Float (slack) per task
  • Critical path
  • Worker allocation timeline
  • Gantt chart (matplotlib)
  • PERT three-point estimates (optimistic/most-likely/pessimistic)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

from task_engine import Task


# ─────────────────────────────────────────────────────────────────────────────
# CPM / PERT data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScheduledTask:
    task:         str
    crew:         str
    duration:     float
    workers:      int
    # CPM results
    es: float = 0.0   # Early Start
    ef: float = 0.0   # Early Finish
    ls: float = 0.0   # Late Start
    lf: float = 0.0   # Late Finish
    tf: float = 0.0   # Total Float
    # PERT
    pert_optimistic:   float = 0.0
    pert_likely:       float = 0.0
    pert_pessimistic:  float = 0.0
    pert_expected:     float = 0.0
    pert_variance:     float = 0.0
    on_critical_path:  bool  = False
    cost_usd:          float = 0.0
    description:       str   = ""

    @property
    def pert_std(self) -> float:
        return math.sqrt(self.pert_variance)

    def to_dict(self) -> dict:
        return {
            "task":               self.task,
            "crew":               self.crew,
            "duration_days":      round(self.duration, 1),
            "es":                 round(self.es, 1),
            "ef":                 round(self.ef, 1),
            "ls":                 round(self.ls, 1),
            "lf":                 round(self.lf, 1),
            "float_days":         round(self.tf, 1),
            "on_critical_path":   self.on_critical_path,
            "pert_expected":      round(self.pert_expected, 1),
            "pert_std":           round(self.pert_std, 2),
            "workers":            self.workers,
            "cost_usd":           round(self.cost_usd),
            "description":        self.description,
        }


@dataclass
class Schedule:
    tasks:            list[ScheduledTask]
    project_duration: float
    critical_path:    list[str]
    total_cost:       float
    pert_p50:         float    # PERT 50th percentile
    pert_p90:         float    # PERT 90th percentile
    worker_peak:      int

    def to_dict(self) -> dict:
        return {
            "project_duration_days": round(self.project_duration, 1),
            "critical_path":         self.critical_path,
            "total_cost_usd":        round(self.total_cost),
            "pert_p50_days":         round(self.pert_p50, 1),
            "pert_p90_days":         round(self.pert_p90, 1),
            "worker_peak":           self.worker_peak,
            "tasks":                 [t.to_dict() for t in self.tasks],
        }

    def summary(self) -> str:
        lines = [
            "═"*58,
            "  PROJECT SCHEDULE — CPM / PERT",
            "═"*58,
            f"  Duration (CPM)  : {self.project_duration:.0f} days",
            f"  Duration (P90)  : {self.pert_p90:.0f} days",
            f"  Total cost      : ${self.total_cost:,.0f}",
            f"  Peak workers    : {self.worker_peak}",
            f"  Critical path   : {' → '.join(self.critical_path[:6])}{'...' if len(self.critical_path)>6 else ''}",
            "═"*58,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class ProjectScheduler:
    """
    CPM + PERT project scheduler.

    Usage
    -----
    sched = ProjectScheduler()
    schedule = sched.compute(tasks, dag)
    sched.gantt_chart(schedule, save_path="gantt.png")
    """

    # PERT spread factors (optimistic = 0.7×, pessimistic = 1.5×)
    PERT_OPT  = 0.70
    PERT_PESS = 1.55

    CREW_COLORS = {
        "civil":       "#4682B4",
        "structural":  "#DC143C",
        "mep":         "#20B2AA",
        "electrical":  "#FFD700",
        "finishing":   "#3CB371",
    }

    def compute(
        self,
        tasks: list[Task],
        dag:   nx.DiGraph,
    ) -> Schedule:

        task_map = {t.task: t for t in tasks}

        # ── PERT three-point estimates ─────────────────────────────────
        pert_map: dict[str, ScheduledTask] = {}
        for t in tasks:
            opt  = t.duration * self.PERT_OPT
            pess = t.duration * self.PERT_PESS
            exp  = (opt + 4*t.duration + pess) / 6
            var  = ((pess - opt) / 6) ** 2
            st   = ScheduledTask(
                task        = t.task,
                crew        = t.crew,
                duration    = t.duration,
                workers     = t.workers,
                cost_usd    = t.cost_usd,
                description = t.description,
                pert_optimistic  = round(opt, 1),
                pert_likely      = round(t.duration, 1),
                pert_pessimistic = round(pess, 1),
                pert_expected    = round(exp, 2),
                pert_variance    = round(var, 4),
            )
            pert_map[t.task] = st

        # ── CPM forward pass (Early Start / Early Finish) ─────────────
        topo = list(nx.topological_sort(dag))
        for name in topo:
            st = pert_map.get(name)
            if st is None:
                continue
            preds = list(dag.predecessors(name))
            st.es = max((pert_map[p].ef for p in preds if p in pert_map), default=0.0)
            st.ef = st.es + st.duration

        project_dur = max((st.ef for st in pert_map.values()), default=0.0)

        # ── CPM backward pass (Late Start / Late Finish) ──────────────
        for name in reversed(topo):
            st = pert_map.get(name)
            if st is None:
                continue
            succs = list(dag.successors(name))
            if not succs:
                st.lf = project_dur
            else:
                st.lf = min((pert_map[s].ls for s in succs if s in pert_map), default=project_dur)
            st.ls = st.lf - st.duration
            st.tf = round(st.lf - st.ef, 4)

        # ── Critical path ─────────────────────────────────────────────
        for st in pert_map.values():
            st.on_critical_path = abs(st.tf) < 0.01

        cp_nodes = [n for n in topo if pert_map.get(n) and pert_map[n].on_critical_path]
        critical_path = cp_nodes

        # ── PERT project statistics ────────────────────────────────────
        cp_expected  = sum(pert_map[n].pert_expected  for n in cp_nodes if n in pert_map)
        cp_variance  = sum(pert_map[n].pert_variance  for n in cp_nodes if n in pert_map)
        cp_std       = math.sqrt(cp_variance) if cp_variance > 0 else 0.0
        pert_p50     = cp_expected
        pert_p90     = cp_expected + 1.28 * cp_std   # normal distribution

        # ── Worker allocation ─────────────────────────────────────────
        days = int(project_dur) + 1
        workers_per_day = np.zeros(days, dtype=int)
        for st in pert_map.values():
            d_start = int(st.es)
            d_end   = min(int(st.ef), days-1)
            workers_per_day[d_start:d_end+1] += st.workers

        peak_workers = int(workers_per_day.max()) if len(workers_per_day) else 0

        total_cost = sum(t.cost_usd for t in tasks)

        scheduled_list = [pert_map[n] for n in topo if n in pert_map]

        return Schedule(
            tasks            = scheduled_list,
            project_duration = round(project_dur, 1),
            critical_path    = critical_path,
            total_cost       = total_cost,
            pert_p50         = round(pert_p50, 1),
            pert_p90         = round(pert_p90, 1),
            worker_peak      = peak_workers,
        )

    # ── Gantt chart ──────────────────────────────────────────────────────

    def gantt_chart(
        self,
        schedule:  Schedule,
        save_path: Optional[str] = None,
        show:      bool = False,
        max_tasks: int  = 35,
    ) -> plt.Figure:

        tasks = schedule.tasks[:max_tasks]
        n     = len(tasks)
        fig_h = max(8, n * 0.32 + 2)

        fig, axes = plt.subplots(
            1, 2, figsize=(18, fig_h),
            gridspec_kw={"width_ratios": [4, 1]},
        )
        fig.patch.set_facecolor("#0a0f1e")

        ax   = axes[0]
        ax_r = axes[1]
        ax.set_facecolor("#0d1b2a")
        ax_r.set_facecolor("#0d1b2a")

        proj_dur = schedule.project_duration

        # Grid lines (every 10 days)
        for d in range(0, int(proj_dur)+1, 10):
            ax.axvline(d, color="#ffffff10", linewidth=0.5)

        # Task bars
        for i, st in enumerate(reversed(tasks)):
            y_pos  = i
            color  = self.CREW_COLORS.get(st.crew, "#888888")
            cp_col = "#FF4444" if st.on_critical_path else color

            # Main bar
            bar = mpatches.FancyBboxPatch(
                (st.es, y_pos - 0.35), st.duration, 0.70,
                boxstyle="round,pad=0.02",
                facecolor=to_rgba(cp_col, 0.85),
                edgecolor="white" if st.on_critical_path else to_rgba("white", 0.3),
                linewidth=1.5 if st.on_critical_path else 0.5,
                zorder=5,
            )
            ax.add_patch(bar)

            # Float bar (grey extension)
            if st.tf > 0.5:
                float_bar = mpatches.FancyBboxPatch(
                    (st.ef, y_pos - 0.15), st.tf, 0.30,
                    boxstyle="round,pad=0.01",
                    facecolor=to_rgba("#888888", 0.3),
                    edgecolor="none",
                    zorder=4,
                )
                ax.add_patch(float_bar)

            # Label
            label_x = st.es + st.duration / 2
            disp_name = st.task.replace("_"," ")
            ax.text(label_x, y_pos, disp_name,
                    ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold",
                    zorder=6, clip_on=True)

        # PERT P90 marker
        ax.axvline(schedule.pert_p90, color="#FFD700", linewidth=1.5,
                   linestyle="--", alpha=0.8, zorder=10)
        ax.text(schedule.pert_p90+0.5, n-0.5, f"P90={schedule.pert_p90:.0f}d",
                color="#FFD700", fontsize=8, va="top")

        ax.set_xlim(0, proj_dur * 1.05)
        ax.set_ylim(-0.8, n)
        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [t.task.replace("_"," ") for t in reversed(tasks)],
            fontsize=7, color="white",
        )
        ax.set_xlabel("Project Days", color="white", fontsize=10)
        ax.set_title("Construction Gantt Chart — CPM / PERT",
                     color="white", fontsize=12, fontweight="bold")
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

        # Legend
        legend_h = [
            mpatches.Patch(color=c, label=crew.title())
            for crew, c in self.CREW_COLORS.items()
        ] + [
            mpatches.Patch(color="#FF4444", label="Critical path"),
            mpatches.Patch(color="#888888", alpha=0.5, label="Float"),
        ]
        ax.legend(handles=legend_h, loc="lower right",
                  facecolor="#1a1a2e", edgecolor="#444",
                  labelcolor="white", fontsize=7)

        # ── Worker allocation chart ────────────────────────────────────
        days = int(proj_dur) + 1
        workers_day = np.zeros(days, dtype=int)
        for st in tasks:
            d0 = int(st.es); d1 = min(int(st.ef), days-1)
            workers_day[d0:d1+1] += st.workers

        ax_r.set_facecolor("#0d1b2a")
        ax_r.barh(range(days)[::-1], workers_day[:days][::-1],
                  color="#4FC3F7", alpha=0.7, height=1.0)
        ax_r.axvline(schedule.worker_peak, color="#FFD700",
                     linewidth=1.5, linestyle="--")
        ax_r.set_xlabel("Workers", color="white", fontsize=8)
        ax_r.set_title("Daily Workers", color="white", fontsize=9, fontweight="bold")
        ax_r.tick_params(colors="white", labelsize=7)
        for sp in ax_r.spines.values():
            sp.set_edgecolor("#333")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=140, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  Gantt chart → {save_path}")
        if show:
            plt.show()
        return fig
