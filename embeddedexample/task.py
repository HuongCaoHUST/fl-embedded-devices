"""Ultralytics YOLO11 helpers used by the Flower clients and server."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from ultralytics import YOLO


def build_model(model_name: str) -> YOLO:
    """Create a YOLO11 model from a model name, YAML file, or checkpoint."""
    return YOLO(model_name)


def get_trainable_state(model: YOLO) -> dict[str, torch.Tensor]:
    """Return floating-point tensors only, which are safe to average with FedAvg."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.model.state_dict().items()
        if tensor.is_floating_point()
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
        name="train",
        exist_ok=True,
        verbose=False,
    )


def evaluate(model: YOLO, data: str, image_size: int, batch_size: int, device: str):
    """Evaluate one client and return its Ultralytics validation metrics."""
    return model.val(
        data=data,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        plots=False,
        verbose=False,
    )
