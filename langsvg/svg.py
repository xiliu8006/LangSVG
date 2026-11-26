"""SVG primitives: mask -> closed Bezier path, colors, and differentiable rendering (diffvg)."""
import copy
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence, Union

import cv2
import numpy as np
import pydiffvg
import torch

from .anchor_span_fit import anchored_vertices
from .curve_fit import contour_vertex_indices, span_fit_points

ParamConf = Union[dict, Callable[[Optional[int], Optional[int]], dict], None]


# ----------------------------------------------------------------------------- geometry
def approx_polygon(contour, epsilon: float, max_edges: int = 8, fallback_epsilon: Optional[float] = None):
    """cv2.approxPolyDP whose tolerance is increased until the polygon has <= max_edges vertices."""
    max_edges = max(3, int(max_edges))
    base = max(0.0, float(epsilon))
    poly = cv2.approxPolyDP(contour, base, closed=True)
    if len(poly) <= max_edges:
        return poly
    fallback = base if fallback_epsilon is None else max(base, float(fallback_epsilon))
    poly = cv2.approxPolyDP(contour, fallback, closed=True)
    if len(poly) <= max_edges:
        return poly

    perimeter = max(float(cv2.arcLength(contour, True)), fallback + 1.0)
    low, high = fallback, min(perimeter, max(fallback * 2.0, fallback + 1.0))
    best, best_eps = None, high
    for _ in range(24):  # exponential search for a feasible tolerance
        candidate = cv2.approxPolyDP(contour, high, closed=True)
        if len(candidate) <= max_edges:
            best, best_eps = candidate, high
            break
        if high >= perimeter:
            break
        high = min(perimeter, max(high * 2.0, high + 1.0))
    if best is None:
        return cv2.approxPolyDP(contour, perimeter, closed=True)
    for _ in range(16):  # bisection for the smallest feasible tolerance
        mid = (low + best_eps) / 2.0
        candidate = cv2.approxPolyDP(contour, mid, closed=True)
        if len(candidate) <= max_edges:
            best, best_eps = candidate, mid
        else:
            low = mid
    return best


def _insert_points(points: np.ndarray, num: int) -> np.ndarray:
    out = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        out.append(p1)
        xs = np.linspace(p1[0], p2[0], num=num + 2)[1:-1]
        ys = np.linspace(p1[1], p2[1], num=num + 2)[1:-1]
        out.extend(np.column_stack((xs, ys)))
    out.append(points[-1])
    return np.array(out)


def _cps_of(conf: dict) -> int:
    cps = int(conf.get("control_points_per_segment", 2))
    if cps < 0 or cps > 2:
        raise ValueError(f"control_points_per_segment must be 0, 1 or 2, got {cps}")
    return cps


def _mask_polygon(mask, epsilon: float, conf: dict):
    """Largest outer contour (M, 2) and its simplified polygon (N, 2); ``on_contour`` is False when
    degenerate vertices had to be invented (then the polygon is not a subsequence of the contour)."""
    if isinstance(mask, torch.Tensor):
        mask = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
    mask = np.ascontiguousarray(np.where(mask > 1, 255, 0).astype(np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    poly = approx_polygon(contour, epsilon, max_edges=int(conf.get("max_polygon_edges", 10)),
                          fallback_epsilon=conf.get("fallback_epsilon"))
    on_contour = len(poly) >= 3
    if len(poly) == 2:  # degenerate: add a third vertex
        new_point = np.array([poly[0, 0, 0] + random.uniform(2, 5), poly[0, 0, 1] + random.uniform(2, 5)])
        poly = np.insert(poly, 1, new_point, axis=0)
    if len(poly) == 1:
        poly = np.insert(poly, 1, np.array([poly[0, 0, 0] + 3, poly[0, 0, 1] + 3]), axis=0)
        poly = np.insert(poly, 1, np.array([poly[0, 0, 0] - 3, poly[0, 0, 1] + 3]), axis=0)
    return contour[:, 0, :], poly[:, 0, :], on_contour


def _straight_points(poly: np.ndarray, cps: int) -> np.ndarray:
    """One segment per polygon edge, control points on the edge."""
    return _insert_points(np.vstack((poly, poly[0])), cps)[:-1]


def _make_path(points: np.ndarray, cps: int) -> pydiffvg.Path:
    return pydiffvg.Path(
        num_control_points=torch.LongTensor([cps] * (len(points) // (cps + 1))),
        points=torch.FloatTensor(np.asarray(points, dtype=np.float32)),
        stroke_width=torch.tensor(0.0),
        is_closed=True,
    )


def mask_to_path(mask, epsilon: float = 5, conf: Optional[dict] = None) -> pydiffvg.Path:
    """Largest outer contour -> simplified polygon -> closed path.

    ``conf`` keys: ``control_points_per_segment`` (0/1/2), ``max_polygon_edges``, ``fallback_epsilon``,
    ``span_fit`` (one segment spans ``cps + 1`` polygon edges, inner control points least-squares fitted
    to the contour; see :mod:`langsvg.curve_fit`), ``span_fit_samples``, ``span_fit_anchor_epsilon``
    (vertices of a second, coarser polygon with this tolerance are always segment endpoints; see
    :mod:`langsvg.anchor_span_fit`).
    Without ``span_fit``: one segment per edge, initialized straight.
    """
    return masks_to_paths([mask], epsilon, [conf or {}])[0]


def masks_to_paths(masks: Sequence, epsilon: float, confs: Sequence[Optional[dict]], workers: int = 8,
                   device: Optional[torch.device] = None) -> List[pydiffvg.Path]:
    """Batched :func:`mask_to_path`. Masks are independent: contours/polygons are computed in a thread pool
    (cv2 releases the GIL), and all span fits of all masks are one batched least-squares solve."""
    confs = [c or {} for c in confs]
    if len(masks) == 0:
        return []
    if len(masks) == 1 or workers <= 1:
        polys = [_mask_polygon(m, epsilon, c) for m, c in zip(masks, confs)]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(masks))) as pool:
            polys = list(pool.map(lambda mc: _mask_polygon(mc[0], epsilon, mc[1]), zip(masks, confs)))

    paths: List[Optional[pydiffvg.Path]] = [None] * len(masks)
    fit_jobs = {}  # cps -> [(index, contour, vertex indices)]
    for i, ((contour, poly, on_contour), conf) in enumerate(zip(polys, confs)):
        cps = _cps_of(conf)
        if conf.get("span_fit") and cps > 0 and on_contour:
            vidx = contour_vertex_indices(contour, poly)
            if vidx is not None:
                anchor = None
                if conf.get("span_fit_anchor_epsilon"):
                    anchored = anchored_vertices(contour, vidx, float(conf["span_fit_anchor_epsilon"]))
                    if anchored is not None:
                        vidx, anchor = anchored
                fit_jobs.setdefault(cps, []).append((i, contour, vidx, conf, anchor))
                continue
        paths[i] = _make_path(_straight_points(poly, cps), cps)
    for cps, jobs in fit_jobs.items():
        fitted = span_fit_points([j[1] for j in jobs], [j[2] for j in jobs], cps,
                                 samples=int(jobs[0][3].get("span_fit_samples", 64)), device=device,
                                 anchors=[j[4] for j in jobs])
        for (i, *_), points in zip(jobs, fitted):
            paths[i] = _make_path(points, cps)
    return paths


def _closest_pair(contour1, contour2):
    pair, best = (None, None), float("inf")
    for outer in contour1:
        for inner in contour2:
            dist = np.linalg.norm(outer[0] - inner[0])
            if dist < best:
                best, pair = dist, (tuple(outer[0]), tuple(inner[0]))
    return pair


def _cv2_point(point):
    if point is None:
        return None
    point = np.asarray(point).reshape(-1)
    if point.size < 2:
        return None
    return int(round(float(point[0]))), int(round(float(point[1])))


def cut_holes_open(mask) -> np.ndarray:
    """Cut a thin channel from every inner contour to the outer one so a single closed path can represent it."""
    if isinstance(mask, torch.Tensor):
        mask = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
    mask = np.ascontiguousarray(np.where(mask > 1, 255, 0).astype(np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    ordered = sorted(zip([cv2.contourArea(c) for c in contours], contours), key=lambda x: x[0], reverse=True)
    if len(contours) > 1:
        for _, contour in ordered[1:]:
            p1, p2 = _closest_pair(ordered[0][1], contour)
            p1, p2 = _cv2_point(p1), _cv2_point(p2)
            if p1 is not None and p2 is not None:
                cv2.line(mask, p1, p2, 0, 3)
    return mask


def point_count(shape) -> int:
    points = getattr(shape, "points", None)
    return 0 if points is None else int(points.shape[0])


def total_points(shapes) -> int:
    return int(sum(point_count(shape) for shape in shapes))


# ----------------------------------------------------------------------------- color
def mean_color(image: np.ndarray, mask: np.ndarray):
    pixels = image[mask > 0]
    return tuple(np.mean(pixels, axis=0, dtype=int)) if len(pixels) > 0 else (0, 0, 0)


def new_group(color_rgb, index: int = 0) -> pydiffvg.ShapeGroup:
    return pydiffvg.ShapeGroup(
        shape_ids=torch.LongTensor([index]),
        fill_color=torch.FloatTensor(list(color_rgb) + [255]) / 255,
        stroke_color=torch.FloatTensor([0, 0, 0, 1]),
    )


def init_svg_from_layers(layers: list, image: np.ndarray, epsilon: float, conf_fn: ParamConf):
    masks, confs = [], []
    for layer_index, layer in enumerate(layers):
        for mask_index, mask in enumerate(layer):
            masks.append(mask)
            confs.append(conf_fn(layer_index, mask_index) if callable(conf_fn) else conf_fn)
    shapes = masks_to_paths(masks, epsilon, confs)
    groups = [pydiffvg.ShapeGroup(
        shape_ids=torch.LongTensor([i]),
        fill_color=torch.tensor(list(mean_color(image, mask)) + [255]) / 255,
        stroke_color=torch.FloatTensor([0, 0, 0, 1]),
    ) for i, mask in enumerate(masks)]
    return shapes, groups


def kmeans_quantize(image: np.ndarray, k: int) -> np.ndarray:
    data = np.float32(image.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return np.uint8(centers)[labels.flatten()].reshape(image.shape)


def dominant_color_fitting(groups, image: np.ndarray, layers: list, cluster: bool, k: int = 30,
                           max_color_change: Optional[float] = None):
    """Set each path color to the dominant color of its visible (not covered by the next layer) region.

    Returns the updated groups and the (optionally k-means quantized) image used for fitting.
    """
    if cluster:
        image = kmeans_quantize(image, k)

    def combine(masks):
        if len(masks) == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        combined = masks[0].copy()
        for mask in masks[1:]:
            combined += mask
        return combined

    dominant = []
    for i, masks in enumerate(layers):
        if len(masks) == 0:
            continue
        covered = combine(layers[i + 1]) if i <= len(layers) - 2 else np.zeros_like(masks[0])
        for mask in masks:
            colors = image[np.where((mask - covered) == 255)]
            counts = Counter(tuple(color) for color in colors)
            dominant.append(list(counts.most_common(1)[0][0]) if counts else [0, 0, 0])
    for i, group in enumerate(groups):
        if max_color_change is not None and getattr(group, "fill_color", None) is not None:
            old = group.fill_color[:3].detach().cpu().numpy().astype(np.float32) * 255.0
            if float(np.sum(np.abs(np.array(dominant[i], dtype=np.float32) - old))) > max_color_change:
                continue
        group.fill_color = torch.FloatTensor(dominant[i] + [255]) / 255
    return groups, image


def clamp_colors(groups) -> None:
    with torch.no_grad():
        for group in groups:
            if getattr(group, "fill_color", None) is not None:
                group.fill_color.data.clamp_(0.0, 1.0)
            if getattr(group, "stroke_color", None) is not None:
                group.stroke_color.data.clamp_(0.0, 1.0)


def reindex(groups) -> None:
    for index, group in enumerate(groups):
        group.shape_ids = torch.LongTensor([index])


# ----------------------------------------------------------------------------- rendering
def render(width: int, height: int, shapes, groups, device) -> torch.Tensor:
    """RGBA image (H, W, 4) in [0, 1]."""
    scene = pydiffvg.RenderFunction.serialize_scene(width, height, shapes, groups)
    return pydiffvg.RenderFunction.apply(width, height, 2, 2, 0, None, *scene).to(device)


def composite(rgba: torch.Tensor, device, background=None) -> torch.Tensor:
    """Alpha-composite over a background color; returns (3, H, W)."""
    if background is None:
        background = torch.tensor([1., 1., 1.], requires_grad=False, device=device)
    rgb = rgba[:, :, 3:4] * rgba[:, :, :3] + background * (1 - rgba[:, :, 3:4])
    return rgb.permute(2, 0, 1)


def render_rgb(width: int, height: int, shapes, groups, device) -> torch.Tensor:
    return composite(render(width, height, shapes, groups, device), device)


def render_subset(width: int, height: int, shapes, groups, indices: List[int], device) -> torch.Tensor:
    if len(indices) == 0:
        return torch.zeros((height, width, 4), dtype=torch.float32, device=device)
    sub_shapes = [copy.deepcopy(shapes[i]) for i in indices]
    sub_groups = []
    for new_index, old_index in enumerate(indices):
        group = copy.deepcopy(groups[old_index])
        group.shape_ids = torch.LongTensor([new_index])
        sub_groups.append(group)
    return render(width, height, sub_shapes, sub_groups, device)


def render_node_mask(width: int, height: int, shapes, groups, index: int, device, visible_only: bool = False):
    """Binary mask of one path, optionally only where it is not covered by later paths."""
    alpha = render_subset(width, height, shapes, groups, [index], device)[:, :, 3]
    if visible_only:
        later = list(range(index + 1, len(shapes)))
        if later:
            alpha = alpha * (1.0 - render_subset(width, height, shapes, groups, later, device)[:, :, 3])
    return np.where(alpha.detach().cpu().numpy() > 1e-3, 255, 0).astype(np.uint8)


def to_uint8_image(rgb_chw: torch.Tensor) -> np.ndarray:
    return np.clip(rgb_chw.permute(1, 2, 0).detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)


def psnr(reference_uint8: np.ndarray, prediction_01: np.ndarray) -> float:
    reference = reference_uint8.astype(np.float32) / 255.0
    prediction = np.clip(prediction_01.astype(np.float32), 0.0, 1.0)
    mse = np.mean((reference - prediction) ** 2)
    return float("inf") if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))
