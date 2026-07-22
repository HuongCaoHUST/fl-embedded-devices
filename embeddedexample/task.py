"""Ultralytics YOLO11 helpers used by the Flower clients and server."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.torch_utils import intersect_dicts

MODEL_PARTS = ("backbone", "neck", "head")


def parse_merge_parts(value) -> tuple[str, ...]:
    """Normalize a YAML list or comma-separated run-config value."""
    raw_parts = value if isinstance(value, (list, tuple)) else str(value).split(",")
    parts = tuple(dict.fromkeys(str(part).strip().lower() for part in raw_parts))
    unknown = set(parts).difference(MODEL_PARTS)
    if not parts or unknown:
        raise ValueError(
            f"merge_parts must contain backbone, neck and/or head; got {value!r}"
        )
    return parts


def load_class_names(dataset_config: str) -> dict[int, str]:
    """Read class names without loading any images from a YOLO data YAML."""
    config = YAML.load(check_yaml(dataset_config))
    names = config["names"]
    return (
        {int(index): str(name) for index, name in names.items()}
        if isinstance(names, dict)
        else {index: str(name) for index, name in enumerate(names)}
    )


def parameter_part(model: YOLO, name: str) -> str:
    """Map a YOLO state_dict tensor to backbone, neck, or detection head."""
    fields = name.split(".")
    if len(fields) < 3 or fields[0] != "model" or not fields[1].isdigit():
        raise ValueError(f"Cannot map YOLO tensor to a model part: {name}")
    module_index = int(fields[1])
    backbone_modules = len(model.model.yaml["backbone"])
    detection_head_index = len(model.model.model) - 1
    if module_index < backbone_modules:
        return "backbone"
    if module_index < detection_head_index:
        return "neck"
    return "head"


def build_model(
    model_name: str, class_names: Mapping[int, str] | None = None
) -> YOLO:
    """Create YOLO, optionally replacing its detection head before FL starts."""
    model = YOLO(model_name)
    detection_head = model.model.model[-1]
    current_num_classes = int(detection_head.nc)
    if class_names is None or len(class_names) == current_num_classes:
        if class_names is not None:
            model.model.names = dict(class_names)
        return model

    pretrained = model.model
    adapted = DetectionModel(
        cfg=pretrained.yaml,
        ch=pretrained.yaml.get("channels", 3),
        nc=len(class_names),
        verbose=False,
    )
    compatible = intersect_dicts(pretrained.state_dict(), adapted.state_dict())
    adapted.load_state_dict(compatible, strict=False)
    adapted.names = dict(class_names)
    model.model = adapted
    return model


def get_trainable_state(
    model: YOLO, merge_parts=None
) -> dict[str, torch.Tensor]:
    """Return floating-point tensors only, which are safe to average with FedAvg."""
    selected = set(parse_merge_parts(merge_parts or MODEL_PARTS))
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.model.state_dict().items()
        if tensor.is_floating_point() and parameter_part(model, name) in selected
    }


def set_trainable_state(model: YOLO, state: Mapping[str, torch.Tensor]) -> None:
    """Merge federated tensors while preserving integer/non-floating buffers."""
    local_state = model.model.state_dict()
    unknown = set(state).difference(local_state)
    if unknown:
        raise ValueError(f"Global model contains unknown tensors: {sorted(unknown)[:5]}")
    for name, tensor in state.items():
        target = local_state[name]
        local_state[name] = tensor.to(device=target.device, dtype=target.dtype)
    model.model.load_state_dict(local_state, strict=True)


def resolve_dataset_config(node_config: Mapping, run_config: Mapping) -> str:
    """Resolve this client's YOLO data YAML (or an Ultralytics built-in YAML)."""
    configured = node_config.get("dataset-config", run_config["dataset-config"])
    partition_id = int(node_config.get("partition-id", 0))
    dataset_config = str(configured).format(partition_id=partition_id)
    if dataset_config.endswith((".yaml", ".yml")) and Path(dataset_config).exists():
        return str(Path(dataset_config).expanduser().resolve())
    return dataset_config


def train(
    model: YOLO,
    data: str,
    epochs: int,
    image_size: int,
    batch_size: int,
    device: str,
    learning_rate: float,
    project: str,
    workers: int = 8,
):
    """Train one client locally and return its Ultralytics results object."""
    return model.train(
        data=data,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        lr0=learning_rate,
        project=project,
        workers=workers,
        name="train",
        exist_ok=True,
        verbose=False,
    )


def evaluate(
    model: YOLO,
    data: str,
    image_size: int,
    batch_size: int,
    device: str,
    workers: int = 8,
):
    """Evaluate one client and return its Ultralytics validation metrics."""
    return model.val(
        data=data,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=workers,
        plots=False,
        verbose=False,
    )
