"""Point budget and depth-adaptive path simplification.

The budget is the total number of path points (``sum(path.points.shape[0])``), e.g. 576, 1152 or 2304 in the
paper. Shape complexity is controlled per tree depth: masks deeper in the tree are fitted with more polygon
vertices, shallow (mostly occluded) ones with fewer.
"""
from typing import Optional

import numpy as np


class Budget:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.max_points = int(cfg["point_budget"])
        # Reference "path count" (one path ~ 9 points); several heuristics scale with it.
        self.path_count = int(cfg.get("path_budget") or self.max_points // 9)
        self.path_cfg = cfg["path"]
        self.cps = int(self.path_cfg["control_points_per_segment"])
        # Paths added during refinement (L1-SAM-Prompting, direct L1, merges) may use a different segment
        # type than the initial tree, e.g. cubic initial tree + quadratic refinement paths.
        refine_cps = self.path_cfg.get("refine_control_points_per_segment")
        self.refine_cps = self.cps if refine_cps is None else int(refine_cps)
        # span fit (langsvg.curve_fit): one segment spans cps + 1 polygon edges -> ~1 path point per vertex;
        # used for the initial tree and the refinement paths
        self.span_fit = bool(self.path_cfg.get("span_fit", False))

    # ------------------------------------------------------------------ global quantities
    def halving_exponent(self) -> float:
        reference = max(1, int(self.path_cfg["reference_path_count"]))
        return float(np.log2(reference / max(1, self.path_count)))

    def epsilon(self) -> int:
        """approxPolyDP tolerance; grows when the budget shrinks."""
        p = self.path_cfg
        return max(1, int(round(float(p["epsilon"]) + float(p["epsilon_step_per_path_halving"]) * self.halving_exponent())))

    def fallback_epsilon(self) -> int:
        return max(1, int(round(self.epsilon() + float(self.path_cfg.get("fallback_epsilon_delta", 4)))))

    def min_points_per_new_path(self) -> int:
        """Smallest new (refinement) path: a triangle."""
        return max(1, 3 * (self.refine_cps + 1))

    def init_limit(self) -> int:
        ratio = min(max(float(self.cfg["init_budget_ratio"]), 0.0), 1.0)
        return self.max_points if ratio <= 0 else max(1, int(self.max_points * ratio))

    def candidates_for(self, remaining_points: int) -> int:
        return max(0, int(remaining_points)) // self.min_points_per_new_path()

    # ------------------------------------------------------------------ per-path complexity
    def max_polygon_edges(self) -> int:
        """Vertex cap of a polygon; grows with the budget."""
        p = self.path_cfg
        min_edges = max(3, int(p.get("min_max_polygon_edges", 3)))
        value = int(round(float(p["max_polygon_edges"]) + float(p["max_polygon_edges_step_per_path_halving"]) * self.halving_exponent()))
        return max(min_edges, value)

    def fallback_layers(self) -> set:
        depth_cfg = self.path_cfg.get("depth_simplification", {})
        if not depth_cfg.get("enabled", True):
            return set()
        return {int(i) for i in depth_cfg.get("fallback_layers", [])}

    def path_conf(self, layer_index: Optional[int] = None, _mask_index: Optional[int] = None,
                  force_fallback: bool = False, refine: bool = False) -> dict:
        """Parameters of :func:`langsvg.svg.mask_to_path` for a node at tree depth ``layer_index``
        (``refine=True``: a path added during refinement)."""
        conf = {"control_points_per_segment": self.refine_cps if refine else self.cps,
                "max_polygon_edges": self.max_polygon_edges()}
        if self.span_fit:
            conf.update(span_fit=True, span_fit_samples=int(self.path_cfg.get("span_fit_samples", 64)),
                        span_fit_anchor_epsilon=self.path_cfg.get("span_fit_anchor_epsilon"))
        if force_fallback or (layer_index is not None and int(layer_index) + 1 in self.fallback_layers()):
            conf["fallback_epsilon"] = self.fallback_epsilon()
        return conf
