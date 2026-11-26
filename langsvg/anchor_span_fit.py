"""Anchored span fit: two-tolerance polygons decide which vertices must be segment endpoints.

The regular span fit (:mod:`langsvg.curve_fit`) walks the polygon of the default tolerance and lets one
segment span ``span`` edges, so it skips vertices regardless of whether they are real corners. Here a second,
coarse polygon is computed from the same contour with a large tolerance (``span_fit_anchor_epsilon``, e.g. 30):

* fine polygon  (default epsilon, ``max_polygon_edges`` cap) -> the candidate vertices, as before;
* coarse polygon (anchor epsilon)                            -> the vertices that are ALWAYS a segment start/end.

The vertex list used for the fit is the union of both (Douglas-Peucker with a larger tolerance normally gives a
subset of the fine vertices; the union also covers the case where the fine tolerance was raised by the edge cap).
Coarse vertices are passed to :func:`langsvg.curve_fit.segment_ranges` as anchors, so between two anchors the
usual span rule applies and a segment never runs over a coarse vertex. No angle-based corner detection is used.

Only used when the path config sets ``span_fit_anchor_epsilon`` (configs/natural_spanfit.yaml: 30).
"""
from typing import Optional, Tuple

import cv2
import numpy as np

from .curve_fit import contour_vertex_indices


def anchored_vertices(contour: np.ndarray, fine_vertex_idx: np.ndarray,
                      anchor_epsilon: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Vertex indices for the span fit and their anchor flags.

    ``contour``: (M, 2) closed contour, ``fine_vertex_idx``: unrolled contour indices of the fine polygon
    (:func:`contour_vertex_indices`). Returns ``(vertex_idx, anchor)``: strictly increasing contour indices
    (span < M) of the union of fine and coarse vertices, and a bool mask of the coarse ones; ``None`` when the
    coarse polygon cannot be matched to the contour (the caller then falls back to the plain span fit).
    """
    m = len(contour)
    coarse = cv2.approxPolyDP(contour.reshape(-1, 1, 2).astype(np.int32), float(anchor_epsilon), closed=True)[:, 0, :]
    coarse_idx = contour_vertex_indices(contour, coarse) if len(coarse) >= 2 else None
    if coarse_idx is None:
        return None
    fine = {int(i) % m for i in fine_vertex_idx}
    anchors = {int(i) % m for i in coarse_idx}
    idx = np.asarray(sorted(fine | anchors), dtype=np.int64)
    flags = np.asarray([int(i) in anchors for i in idx], dtype=bool)
    return idx, flags
