"""Span fitting: polygon -> Bezier path where one segment spans several polygon edges.

With ``control_points_per_segment = c`` (1: quadratic, 2: cubic) a segment starts at polygon vertex v_i and
ends at v_{i+c+1}, i.e. it spans ``c + 1`` edges and the skipped vertices are replaced by its control points
(one path point per polygon vertex). The last segment ends at v_0; it may span fewer edges. The endpoints are
fixed, the ``c`` inner control points are the least-squares fit to the mask contour between them.

Anchors (``anchor``, see :mod:`langsvg.anchor_span_fit`): vertices that are always a segment start/end; a
segment ends early at an anchor (it may then span fewer edges), and the walk starts at the first anchor so the
closing segment also ends at one.

Everything after cv2 (contour + approxPolyDP, per mask, in a thread pool) is batched over all segments of all
masks: each segment's contour arc is resampled to ``K`` points by arc length -> one (S, K, 2) tensor, and the
least-squares solve is a single batched 2x2 (cubic) / 1x1 (quadratic) normal-equation solve.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

# Bernstein bases of the free control points, by degree.
def _inner_basis(t: torch.Tensor, degree: int) -> torch.Tensor:
    """(..., K) -> (..., K, degree - 1): bases of the inner control points."""
    s = 1.0 - t
    if degree == 2:
        return (2 * s * t).unsqueeze(-1)
    return torch.stack((3 * s * s * t, 3 * s * t * t), dim=-1)


def _end_basis(t: torch.Tensor, degree: int) -> Tuple[torch.Tensor, torch.Tensor]:
    s = 1.0 - t
    return s ** degree, t ** degree


def contour_vertex_indices(contour: np.ndarray, poly: np.ndarray) -> Optional[np.ndarray]:
    """Unrolled (strictly increasing, span < len(contour)) contour indices of the polygon vertices, or None.

    approxPolyDP returns a subsequence of the contour, so each vertex is matched to its next occurrence
    after the previous vertex (a pixel can occur several times on thin parts of a CHAIN_APPROX_NONE contour).
    """
    m = len(contour)
    lookup = {}
    for i, (x, y) in enumerate(contour):
        lookup.setdefault((int(x), int(y)), []).append(i)
    out, prev = [], None
    for x, y in poly:
        occ = lookup.get((int(x), int(y)))
        if not occ:
            return None
        if prev is None:
            idx = occ[0]
        else:
            later = [i for i in occ if i > prev % m]
            base = prev - prev % m
            idx = base + (later[0] if later else occ[0] + m)
        out.append(idx)
        prev = idx
    out = np.asarray(out, dtype=np.int64)
    if len(out) < 2 or np.any(np.diff(out) <= 0) or out[-1] - out[0] >= m:
        return None
    return out


def segment_ranges(vertex_idx: np.ndarray, m: int, span: int, anchor: Optional[np.ndarray] = None) -> np.ndarray:
    """(n_seg, 2) unrolled [start, end] contour indices of the segments; the last one closes at the start + m.

    A segment ends after ``span`` edges, or earlier at an anchor vertex (``anchor``: (N,) bool).
    With anchors the walk starts at the first one, so the closing segment also ends at an anchor.
    """
    n = len(vertex_idx)
    if anchor is not None and anchor.any():
        k = int(np.flatnonzero(anchor)[0])
        vertex_idx = np.concatenate((vertex_idx[k:], vertex_idx[:k] + m))
        anchor = np.roll(anchor, -k)
    else:
        anchor = np.zeros(n, dtype=bool)
    starts, i = [0], 0
    while True:
        j = i + 1
        while j < n and j - i < span and not anchor[j]:
            j += 1
        if j >= n:
            break
        starts.append(j)
        i = j
    starts = vertex_idx[starts]
    ends = np.append(starts[1:], vertex_idx[0] + m)
    return np.stack((starts, ends), axis=1)


def resample_arcs(contours: Sequence[np.ndarray], ranges: Sequence[np.ndarray], samples: int,
                  device: torch.device) -> torch.Tensor:
    """Arc-length resampling of every contour arc [start, end] to ``samples`` points -> (S, K, 2).

    Arcs are padded to the longest one with their last point (zero-length steps), so a single batched
    ``searchsorted`` handles all of them.
    """
    seg_contour, seg_start, seg_len, offsets, flat = [], [], [], [], []
    offset = 0
    for ci, (contour, rng) in enumerate(zip(contours, ranges)):
        flat.append(contour)
        for s, e in rng:
            seg_contour.append(offset)
            seg_start.append(int(s))
            seg_len.append(int(e - s) + 1)
            offsets.append(len(contour))
        offset += len(contour)
    flat = torch.as_tensor(np.concatenate(flat, axis=0), dtype=torch.float32, device=device)
    base = torch.as_tensor(seg_contour, device=device)
    start = torch.as_tensor(seg_start, device=device)
    length = torch.as_tensor(seg_len, device=device)
    m = torch.as_tensor(offsets, device=device)
    lmax = int(length.max())
    step = torch.arange(lmax, device=device).unsqueeze(0)
    step = torch.minimum(step, (length - 1).unsqueeze(1))                  # pad with the last point
    chain = flat[base.unsqueeze(1) + (start.unsqueeze(1) + step) % m.unsqueeze(1)]   # (S, Lmax, 2)

    seglen = (chain[:, 1:] - chain[:, :-1]).norm(dim=-1)
    cum = torch.cat((torch.zeros_like(seglen[:, :1]), seglen.cumsum(1)), dim=1)       # (S, Lmax)
    total = cum[:, -1:].clamp_min(1e-6)
    target = torch.linspace(0, 1, samples, device=device).unsqueeze(0) * total        # (S, K)
    hi = torch.searchsorted(cum.contiguous(), target.contiguous(), right=True).clamp(1, lmax - 1)
    lo = hi - 1
    c_lo, c_hi = cum.gather(1, lo), cum.gather(1, hi)
    w = ((target - c_lo) / (c_hi - c_lo).clamp_min(1e-6)).clamp(0, 1).unsqueeze(-1)
    p_lo = chain.gather(1, lo.unsqueeze(-1).expand(-1, -1, 2))
    p_hi = chain.gather(1, hi.unsqueeze(-1).expand(-1, -1, 2))
    out = p_lo + w * (p_hi - p_lo)
    out[:, 0], out[:, -1] = chain[:, 0], chain[torch.arange(len(chain), device=device), length - 1]
    return out


def fit_inner_controls(samples: torch.Tensor, degree: int, ridge: float = 1e-6) -> torch.Tensor:
    """Least-squares inner control points of Bezier segments with fixed endpoints.

    ``samples``: (S, K, 2), first/last = endpoints, uniform arc-length parameterization. Returns (S, degree - 1, 2).
    """
    S, K, _ = samples.shape
    p0, p3 = samples[:, :1], samples[:, -1:]
    t = torch.linspace(0, 1, K, device=samples.device, dtype=samples.dtype).expand(S, K)
    eye = torch.eye(degree - 1, device=samples.device, dtype=samples.dtype)
    straight = torch.stack([p0[:, 0] + (p3[:, 0] - p0[:, 0]) * (i / degree) for i in range(1, degree)], dim=1)
    a = _inner_basis(t, degree)                                       # (S, K, d-1)
    b0, b3 = _end_basis(t, degree)
    rhs = samples - b0.unsqueeze(-1) * p0 - b3.unsqueeze(-1) * p3     # (S, K, 2)
    ata = a.transpose(1, 2) @ a                                       # (S, d-1, d-1)
    # small ridge towards the straight initialization keeps degenerate arcs (straight, tiny) well posed
    lam = ridge * K
    return torch.linalg.solve(ata + lam * eye, a.transpose(1, 2) @ rhs + lam * straight)


def span_fit_points(contours: List[np.ndarray], vertex_idx: List[np.ndarray], cps: int, samples: int = 64,
                    device: Optional[torch.device] = None,
                    anchors: Optional[List[Optional[np.ndarray]]] = None) -> List[np.ndarray]:
    """Batched span fit of many polygons. Returns per polygon the (n_seg * (cps + 1), 2) path points
    (closed-path layout: anchor, inner controls, next anchor, ...). ``anchors``: per polygon, (N,) bool of
    the vertices that must be segment endpoints (or None)."""
    if not contours:
        return []
    degree = cps + 1
    device = device or torch.device("cpu")
    anchors = anchors or [None] * len(contours)
    ranges = [segment_ranges(v, len(c), cps + 1, a) for c, v, a in zip(contours, vertex_idx, anchors)]
    arcs = resample_arcs(contours, ranges, max(int(samples), degree + 2), device)
    ctrl = fit_inner_controls(arcs, degree)                                # (S, cps, 2)
    pts = torch.cat((arcs[:, :1], ctrl), dim=1).reshape(-1, 2).cpu().numpy()   # anchor + inner per segment
    out, k = [], 0
    for rng in ranges:
        n = len(rng) * (cps + 1)
        out.append(pts[k:k + n])
        k += n
    return out
