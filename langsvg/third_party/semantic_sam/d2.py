"""Minimal stand-ins for the few detectron2 / fvcore utilities used by Semantic-SAM.

Semantic-SAM only needs these at inference time. The implementations below are
copied (and trimmed) from detectron2 (Apache-2.0) and fvcore (Apache-2.0) so the
numerics are identical, while removing detectron2 as an installation dependency.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# ----------------------------------------------------------------------------- registry
class Registry:
    def __init__(self, name: str):
        self._name = name
        self._obj_map: Dict[str, Any] = {}

    def register(self, obj=None):
        if obj is None:
            def deco(func_or_class):
                self._obj_map[func_or_class.__name__] = func_or_class
                return func_or_class
            return deco
        self._obj_map[obj.__name__] = obj
        return obj

    def get(self, name: str):
        if name not in self._obj_map:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return self._obj_map[name]

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map


BACKBONE_REGISTRY = Registry("BACKBONE")
SEM_SEG_HEADS_REGISTRY = Registry("SEM_SEG_HEADS")


# ----------------------------------------------------------------------------- modeling
@dataclass
class ShapeSpec:
    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def size_divisibility(self) -> int:
        return 0

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }


# ----------------------------------------------------------------------------- layers
class Conv2d(torch.nn.Conv2d):
    """nn.Conv2d with optional `norm` and `activation` (detectron2.layers.Conv2d)."""

    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DeformConv(nn.Module):  # imported but never instantiated by Semantic-SAM-L
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DeformConv is not available without detectron2.")


def get_norm(norm, out_channels):
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
            "SyncBN": nn.SyncBatchNorm,
            "GN": lambda channels: nn.GroupNorm(32, channels),
        }[norm]
    return norm(out_channels)


def shapes_to_tensor(x: List[int], device: Optional[torch.device] = None) -> torch.Tensor:
    return torch.as_tensor(x, device=device)


def cat(tensors: List[torch.Tensor], dim: int = 0):
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def point_sample(input, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
    if num_random_points > 0:
        point_coords = cat(
            [point_coords, torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device)],
            dim=1,
        )
    return point_coords


def c2_xavier_fill(module: nn.Module) -> None:
    """fvcore.nn.weight_init.c2_xavier_fill"""
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


# ----------------------------------------------------------------------------- structures
class ImageList:
    def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx) -> torch.Tensor:
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]

    def to(self, *args, **kwargs) -> "ImageList":
        return ImageList(self.tensor.to(*args, **kwargs), self.image_sizes)

    @property
    def device(self):
        return self.tensor.device

    @staticmethod
    def from_tensors(tensors: List[torch.Tensor], size_divisibility: int = 0, pad_value: float = 0.0) -> "ImageList":
        assert len(tensors) > 0 and isinstance(tensors, (tuple, list))
        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
        max_size = torch.stack([shapes_to_tensor(x) for x in image_sizes]).max(0).values
        if size_divisibility > 1:
            stride = size_divisibility
            max_size = (max_size + (stride - 1)).div(stride, rounding_mode="floor") * stride
        if len(tensors) == 1:
            image_size = image_sizes[0]
            padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
            batched_imgs = F.pad(tensors[0], padding_size, value=pad_value).unsqueeze_(0)
        else:
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
            batched_imgs = tensors[0].new_full(batch_shape, pad_value)
            for img, pad_img in zip(tensors, batched_imgs):
                pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
        return ImageList(batched_imgs.contiguous(), image_sizes)


class Boxes:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = torch.as_tensor(tensor, dtype=torch.float32).reshape(-1, 4)

    def to(self, device):
        return Boxes(self.tensor.to(device))

    def __len__(self):
        return self.tensor.shape[0]


class BitMasks:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = torch.as_tensor(tensor).to(torch.bool)
        self.image_size = self.tensor.shape[1:]

    def get_bounding_boxes(self) -> Boxes:
        boxes = torch.zeros(self.tensor.shape[0], 4, dtype=torch.float32)
        x_any = torch.any(self.tensor, dim=1)
        y_any = torch.any(self.tensor, dim=2)
        for idx in range(self.tensor.shape[0]):
            x = torch.where(x_any[idx, :])[0]
            y = torch.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                boxes[idx, :] = torch.as_tensor([x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32)
        return Boxes(boxes)

    def __len__(self):
        return self.tensor.shape[0]


class Instances:
    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self):
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError(f"Cannot find field '{name}' in the given Instances!")
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        self._fields[name] = value

    def has(self, name: str) -> bool:
        return name in self._fields

    def get(self, name: str) -> Any:
        return self._fields[name]


class ROIMasks:  # only used by training-time post-processing
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor


# ----------------------------------------------------------------------------- misc
class _Metadata(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def set(self, **kwargs):
        self.update(kwargs)
        return self


class _MetadataCatalog:
    def __init__(self):
        self._store: Dict[str, _Metadata] = {}

    def get(self, name: str) -> _Metadata:
        return self._store.setdefault(name, _Metadata(name=name))


MetadataCatalog = _MetadataCatalog()


def retry_if_cuda_oom(func):
    return func


def is_main_process() -> bool:
    return True


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


class PathManager:
    open = staticmethod(open)
