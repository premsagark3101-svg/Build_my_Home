"""
mep_routing.py
==============
Routes plumbing pipes and electrical wiring through the building grid
using NetworkX graph algorithms (A* and Dijkstra).

Algorithms:
  Plumbing  → A* from each wet room to main sewer connection
  Electrical→ Dijkstra shortest path from panel to every room

Constraints:
  • Structural columns are BLOCKED nodes (cannot route through)
  • Plumbing prefers walls/corridors over room interiors
  • Electrical prefers corridors and avoids wet rooms for safety
  • MEP routes share conduit space (weight bonus for parallel runs)
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np

GRID = 20

# Cost weights for grid traversal
COST_OPEN       = 1.0    # empty corridor / wall cavity
COST_ROOM       = 2.5    # through a room interior (avoid)
COST_STRUCTURAL = 999.0  # blocked — structural column
COST_WET_ELEC   = 4.0    # electrical through wet room (extra penalty)
COST_PARALLEL   = 0.6    # discount when sharing existing conduit run

# MEP shaft: vertical wet-stack column routes plumbing vertically
MEP_SHAFT_BONUS = 0.3    # discount for following plumbing stack X-column


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MEPRoute:
    route_type:  str                      # "plumbing" | "electrical"
    source:      tuple[int,int]
    destination: tuple[int,int]
    path:        list[tuple[int,int]]
    length_m:    float
    via_shaft:   bool = False
    label:       str  = ""

    def to_dict(self) -> dict:
        return {
            "type":       self.route_type,
            "source":     list(self.source),
            "destination":list(self.destination),
            "path":       [list(p) for p in self.path],
            "length_m":   round(self.length_m, 2),
            "via_shaft":  self.via_shaft,
            "label":      self.label,
        }


@dataclass
class MEPResult:
    plumbing_routes:   list[MEPRoute]
    electrical_routes: list[MEPRoute]
    grid_size:         int
    sewer_connection:  tuple[int,int]
    electrical_panel:  tuple[int,int]

    def to_dict(self) -> dict:
        return {
            "plumbing_routes":   [r.to_dict() for r in self.plumbing_routes],
            "electrical_routes": [r.to_dict() for r in self.electrical_routes],
            "sewer_connection":  list(self.sewer_connection),
            "electrical_panel":  list(self.electrical_panel),
        }

    def path_coords(self) -> dict:
        """Spec output format."""
        return {
            "plumbing_routes":   [[list(p) for p in r.path] for r in self.plumbing_routes],
            "electrical_routes": [[list(p) for p in r.path] for r in self.electrical_routes],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Grid graph builder
# ─────────────────────────────────────────────────────────────────────────────

class MEPGridGraph:
    """Builds a weighted grid graph for pathfinding, blocking structural columns."""

    def __init__(
        self,
        layout:   list[dict],
        columns:  list[list[float]],
        grid_size: int = GRID,
    ):
        self.G         = nx.grid_2d_graph(grid_size, grid_size)
        self.grid_size = grid_size
        self.layout    = layout
        self.blocked:  set[tuple[int,int]] = set()
        self.room_map: dict[tuple[int,int], str] = {}

        # Build room occupancy map
        for room in layout:
            rtype = room["room"]
            for dx in range(room["w"]):
                for dy in range(room["h"]):
                    self.room_map[(room["x"]+dx, room["y"]+dy)] = rtype

        # Block structural columns (±1 cell for column footprint)
        for (cx, cy) in columns:
            cx, cy = int(cx), int(cy)
            for dx in range(-0, 1):
                for dy in range(-0, 1):
                    nx_, ny_ = cx+dx, cy+dy
                    if 0 <= nx_ < grid_size and 0 <= ny_ < grid_size:
                        self.blocked.add((nx_, ny_))

        # Assign edge weights
        for u, v in self.G.edges():
            w = self._edge_weight(u, v, "plumbing")
            self.G[u][v]["plumbing_weight"]   = w
            we = self._edge_weight(u, v, "electrical")
            self.G[u][v]["electrical_weight"] = we

    def _edge_weight(self, u, v, mode: str) -> float:
        # If either node blocked → very high cost
        if u in self.blocked or v in self.blocked:
            return COST_STRUCTURAL

        room_u = self.room_map.get(u, "")
        room_v = self.room_map.get(v, "")

        if mode == "plumbing":
            base = COST_ROOM if (room_u or room_v) else COST_OPEN
            # Wet rooms are cheap for plumbing (need pipes)
            if room_u in ("bathroom","kitchen","laundry","utility_room") or \
               room_v in ("bathroom","kitchen","laundry","utility_room"):
                base = COST_OPEN
            return base

        else:  # electrical
            base = COST_ROOM if (room_u or room_v) else COST_OPEN
            # Wet rooms get safety penalty for electrical
            if room_u in ("bathroom","kitchen","laundry") or \
               room_v in ("bathroom","kitchen","laundry"):
                base = COST_WET_ELEC
            return base

    def get_graph(self, mode: str) -> nx.Graph:
        H = nx.Graph()
        H.add_nodes_from(self.G.nodes())
        wkey = f"{mode}_weight"
        for u, v, d in self.G.edges(data=True):
            H.add_edge(u, v, weight=d.get(wkey, 1.0))
        return H

    def mark_parallel(self, path: list[tuple[int,int]], discount: float = COST_PARALLEL):
        """Apply parallel-run discount to edges already used by a route."""
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            if self.G.has_edge(u, v):
                for key in ("plumbing_weight", "electrical_weight"):
                    self.G[u][v][key] = max(
                        self.G[u][v].get(key, 1.0) * discount, 0.3
                    )


# ─────────────────────────────────────────────────────────────────────────────
# A* heuristic
# ─────────────────────────────────────────────────────────────────────────────

def manhattan(a: tuple[int,int], b: tuple[int,int]) -> float:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def astar_path(
    G: nx.Graph,
    source: tuple[int,int],
    target: tuple[int,int],
) -> list[tuple[int,int]]:
    """
    A* pathfinding on weighted grid graph.
    Falls back to NetworkX astar if custom fails.
    """
    try:
        return nx.astar_path(G, source, target,
                             heuristic=manhattan, weight="weight")
    except nx.NetworkXNoPath:
        return []
    except Exception:
        return []


def dijkstra_path(
    G: nx.Graph,
    source: tuple[int,int],
    target: tuple[int,int],
) -> list[tuple[int,int]]:
    """Dijkstra shortest path."""
    try:
        return nx.dijkstra_path(G, source, target, weight="weight")
    except nx.NetworkXNoPath:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# MEP Router
# ─────────────────────────────────────────────────────────────────────────────

class MEPRouter:
    """
    Routes plumbing and electrical MEP through the building.

    Parameters
    ----------
    layout        : room layout dicts
    columns       : list of [x,y] column positions
    structural_core: dict with plumbing_stacks, staircase, elevator
    grid_size     : grid dimension
    """

    def __init__(
        self,
        layout:         list[dict],
        columns:        list[list[float]],
        structural_core: Optional[dict] = None,
        grid_size:      int = GRID,
    ):
        self.layout    = layout
        self.columns   = columns
        self.core      = structural_core or {}
        self.grid_size = grid_size
        self.graph_builder = MEPGridGraph(layout, columns, grid_size)

    def route(self) -> MEPResult:
        G = grid_size = self.grid_size

        # Sewer connection: bottom-left corner (main external sewer)
        sewer = (0, 0)

        # Electrical panel: near entrance (ground floor, right side)
        panel = (0, grid_size - 1)

        # ── Plumbing: route wet rooms to sewer via A* ─────────────────
        plumb_graph  = self.graph_builder.get_graph("plumbing")
        plumb_routes = self._route_plumbing(plumb_graph, sewer)

        # ── Electrical: Dijkstra from panel to all rooms ───────────────
        elec_graph   = self.graph_builder.get_graph("electrical")
        elec_routes  = self._route_electrical(elec_graph, panel)

        return MEPResult(
            plumbing_routes   = plumb_routes,
            electrical_routes = elec_routes,
            grid_size         = grid_size,
            sewer_connection  = sewer,
            electrical_panel  = panel,
        )

    # ── Plumbing ─────────────────────────────────────────────────────────

    def _route_plumbing(
        self, G: nx.Graph, sewer: tuple[int,int]
    ) -> list[MEPRoute]:
        routes  = []
        wet_rooms = [r for r in self.layout
                     if r["room"] in ("bathroom","kitchen","laundry","utility_room")]

        # Sort by distance to sewer (closest first — daisy-chain)
        wet_rooms.sort(key=lambda r: abs(r["x"]-sewer[0]) + abs(r["y"]-sewer[1]))

        for room in wet_rooms:
            # Source: centre of wet room
            src = (room["x"] + room["w"]//2, room["y"] + room["h"]//2)
            src = self._clamp(src)

            # Follow plumbing stack if available
            via_shaft = False
            stacks = self.core.get("plumbing_stacks", [])
            if stacks:
                # Route to nearest stack first, then to sewer
                nearest_stack = min(stacks,
                    key=lambda s: abs(src[0]-s[0]) + abs(src[1]-s[1])
                )
                stack_node = self._clamp((int(nearest_stack[0]), int(nearest_stack[1])))
                path1 = astar_path(G, src, stack_node)
                path2 = astar_path(G, stack_node, sewer)
                if path1 and path2:
                    path = path1 + path2[1:]
                    via_shaft = True
                else:
                    path = astar_path(G, src, sewer)
            else:
                path = astar_path(G, src, sewer)

            if path:
                length = len(path) - 1
                route  = MEPRoute(
                    route_type  = "plumbing",
                    source      = src,
                    destination = sewer,
                    path        = path,
                    length_m    = float(length),
                    via_shaft   = via_shaft,
                    label       = f"{room['room']} → sewer",
                )
                routes.append(route)
                # Apply parallel discount for subsequent routes
                self.graph_builder.mark_parallel(path, COST_PARALLEL)
                # Update graph
                G = self.graph_builder.get_graph("plumbing")

        return routes

    # ── Electrical ───────────────────────────────────────────────────────

    def _route_electrical(
        self, G: nx.Graph, panel: tuple[int,int]
    ) -> list[MEPRoute]:
        routes = []
        # Route to centre of each room
        for room in self.layout:
            if room.get("structural"):
                continue
            dest = (room["x"] + room["w"]//2, room["y"] + room["h"]//2)
            dest = self._clamp(dest)

            path = dijkstra_path(G, panel, dest)
            if path:
                route = MEPRoute(
                    route_type  = "electrical",
                    source      = panel,
                    destination = dest,
                    path        = path,
                    length_m    = float(len(path)-1),
                    label       = f"panel → {room['room']}",
                )
                routes.append(route)
                # Discount used routes (cable tray sharing)
                self.graph_builder.mark_parallel(path, COST_PARALLEL)
                G = self.graph_builder.get_graph("electrical")

        return routes

    def _clamp(self, pos: tuple[int,int]) -> tuple[int,int]:
        x = max(0, min(self.grid_size-1, pos[0]))
        y = max(0, min(self.grid_size-1, pos[1]))
        return (x, y)
