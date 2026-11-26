"""Differentiable optimization of SVG paths with diffvg.

* :func:`optimize_structure` - fits every tree layer to its masks (structure loss) while
  matching the image; used for the initial tree and for L1-SAM-Prompting proposals.
* :func:`optimize_visual` - plain reconstruction loss with per-path learning-rate scales.
"""
from typing import List, Optional

import pydiffvg
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .svg import clamp_colors, composite, render


def _adam(param_groups: dict):
    groups = [{"params": g["params"], "lr": g["lr"], "_id": name}
              for name, g in sorted(param_groups.items()) if len(g["params"]) > 0]
    return torch.optim.Adam(groups, betas=(0.9, 0.9), eps=1e-6)


def _add(param_groups: dict, name: str, tensor, lr: float) -> None:
    param_groups.setdefault(name, {"params": [], "lr": lr})["params"].append(tensor)


def make_optimizer(shapes, groups, lr: dict, train_color: bool = True,
                   point_scales: Optional[List[float]] = None, color_scales: Optional[List[float]] = None,
                   opt_flags: Optional[List[int]] = None):
    """Adam over path points and fill colors; a scale of 0 freezes the parameter."""
    if not opt_flags:
        opt_flags = [1] * len(shapes)
    params = {}
    for i, path in enumerate(shapes):
        path.id = i
        scale = point_scales[i] if point_scales is not None else (1.0 if opt_flags[i] == 1 else 0.0)
        path.points.requires_grad = scale > 0
        if scale > 0:
            _add(params, f"point_{scale}", path.points, lr["point"] * scale)
    if train_color:
        for i, group in enumerate(groups):
            scale = color_scales[i] if color_scales is not None else (1.0 if opt_flags[i] == 1 else 0.0)
            group.fill_color.requires_grad = scale > 0
            if scale > 0:
                _add(params, f"color_{scale}", group.fill_color, lr["color"] * scale)
    return _adam(params)


def _incremental_optimizer(shapes, groups, new_flags: List[int], lr: dict, train_color: bool = True,
                           old_lr_scale: float = 0.2):
    """New paths: points + colors trainable; old paths: colors only (at ``old_lr_scale``)."""
    params = {}
    for i, path in enumerate(shapes):
        path.id = i
        is_new = new_flags[i] == 1
        path.points.requires_grad = is_new
        if is_new:
            _add(params, "point_new", path.points, lr["point"])
    if train_color:
        for i, group in enumerate(groups):
            group.fill_color.requires_grad = True
            if new_flags[i] == 1:
                _add(params, "color_new", group.fill_color, lr["color"])
            else:
                _add(params, "color_old", group.fill_color, lr["color"] * old_lr_scale)
    return _adam(params)


def _opacity_loss(groups, conf: dict, device):
    """Pull the alpha of trainable fill colors towards ``target`` (keeps paths opaque)."""
    if not conf.get("enabled", False) or float(conf.get("weight", 0.0)) <= 0:
        return torch.tensor(0.0, device=device)
    alphas = [g.fill_color[3] for g in groups
              if getattr(g, "fill_color", None) is not None and g.fill_color.numel() >= 4
              and (not conf.get("trainable_only", True) or g.fill_color.requires_grad)]
    if len(alphas) == 0:
        return torch.tensor(0.0, device=device)
    alpha = torch.stack(alphas).to(device)
    return F.mse_loss(alpha, torch.full_like(alpha, float(conf.get("target", 1.0)))) * float(conf["weight"])


def _exclude_loss(rgb, scale):
    """Penalize pixels that are covered by several semi-transparent paths of the same layer."""
    return torch.sum(F.relu(178 / 255 - rgb)) * scale


def _masked_loss(rgba: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Region-only reconstruction (object extraction, :mod:`langsvg.object_tool`): premultiplied color
    vs. ``target * mask`` plus alpha vs. ``mask`` (paths must stay transparent outside the region)."""
    alpha = rgba[:, :, 3:4]
    pred = (rgba[:, :, :3] * alpha).permute(2, 0, 1)
    return F.mse_loss(pred, target * loss_mask[None]) + F.mse_loss(alpha[:, :, 0], loss_mask)


def _structure_targets(layers: list):
    """Random-colored segmentation image per layer (the structure target of Layered Vectorization)."""
    targets, colors = [], []
    for masks in layers:
        seg, layer_colors = 0, []
        for mask in masks:
            h, w = mask.shape[:2]
            colored = torch.zeros((3, h, w))
            color = []
            for channel in range(3):
                value = 0.2 + (1 - 0.2) * torch.rand(1)
                colored[channel, :, :] = value
                color.append(value)
            layer_colors.append(color)
            seg = torch.clamp(seg + colored * transforms.ToTensor()(Image.fromarray(mask)), max=1.0)
        targets.append(seg)
        colors.append(layer_colors)
    return targets, colors


def optimize_structure(shapes, groups, target_img, layers: list, num_iters: int, lr: dict, device,
                       opacity_conf: dict, new_flags: Optional[List[int]] = None, train_color: bool = True,
                       loss_mask: Optional[torch.Tensor] = None):
    """Structure optimization (``new_flags=None``) or incremental structure optimization.

    loss = w * MSE(render, image) + sum_layers [MSE(layer render, layer segmentation) + exclusion]
    with w = 0.1 (initial tree) or 0.2 (incremental, only new paths move).
    ``loss_mask`` (H, W) in {0, 1}: the MSE term only covers that region (see :func:`_masked_loss`);
    ``None`` (full-image vectorization) keeps the plain loss.
    """
    seg_targets, seg_colors = _structure_targets(layers)
    seg_targets = [t.to(device) for t in seg_targets]
    seg_groups = [[pydiffvg.ShapeGroup(shape_ids=torch.LongTensor([i]), fill_color=torch.FloatTensor(c + [1]),
                                       stroke_color=torch.FloatTensor([0, 0, 0, 1])) for i, c in enumerate(colors)]
                  for colors in seg_colors]
    transparent = [pydiffvg.ShapeGroup(shape_ids=torch.LongTensor([i]), fill_color=torch.FloatTensor([0, 0, 0, 0.3]),
                                       stroke_color=torch.FloatTensor([0, 0, 0, 0.3])) for i in range(len(shapes))]
    black = torch.tensor([0., 0., 0.], requires_grad=False, device=device)
    white = torch.tensor([1., 1., 1.], requires_grad=False, device=device)
    h, w = target_img.shape[:2]
    target = (torch.tensor(target_img, device=device) / 255).permute(2, 0, 1)

    if new_flags is None:
        optimizer, mse_weight = make_optimizer(shapes, groups, lr, train_color=train_color), 0.1
    else:
        optimizer, mse_weight = _incremental_optimizer(shapes, groups, new_flags, lr, train_color), 0.2

    for _ in tqdm(range(num_iters), desc="structure", leave=False):
        loss_struct, loss_exclude, end = 0, 0, 0
        for layer_i, layer_target in enumerate(seg_targets):
            n = len(layers[layer_i])
            end += n
            layer_shapes = shapes[end - n:end]
            seg = composite(render(w, h, layer_shapes, seg_groups[layer_i], device), device, black)
            loss_struct += F.mse_loss(seg, layer_target)
            overlap = composite(render(w, h, layer_shapes, transparent[:n], device), device, white)
            loss_exclude += _exclude_loss(overlap, 2e-7)
        if loss_mask is None:
            img = composite(render(w, h, shapes, groups, device), device, white)
            recon = F.mse_loss(img, target)
        else:
            recon = _masked_loss(render(w, h, shapes, groups, device), target, loss_mask)
        loss = recon * mse_weight + loss_exclude + loss_struct + _opacity_loss(groups, opacity_conf, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        clamp_colors(groups)
    return shapes, groups


def optimize_visual(shapes, groups, target_img, num_iters: int, lr: dict, device,
                    point_scales: List[float], color_scales: List[float], train_color: bool = True,
                    loss_mask: Optional[torch.Tensor] = None):
    """Reconstruction optimization with per-path learning-rate scales (``loss_mask``: region-only loss)."""
    h, w = target_img.shape[:2]
    target = (torch.tensor(target_img, device=device) / 255).permute(2, 0, 1)
    optimizer = make_optimizer(shapes, groups, lr, train_color=train_color,
                               point_scales=point_scales, color_scales=color_scales)
    for _ in tqdm(range(num_iters), desc="visual", leave=False):
        if loss_mask is None:
            loss = F.mse_loss(composite(render(w, h, shapes, groups, device), device), target)
        else:
            loss = _masked_loss(render(w, h, shapes, groups, device), target, loss_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        clamp_colors(groups)
    return shapes, groups
