"""Flower 1.10 client for YOLO training on JetPack 4 devices."""

from __future__ import annotations

import argparse
import ctypes
import gc
from collections.abc import Mapping
from pathlib import Path

import flwr as fl
import numpy as np
import torch
import yaml

from embeddedexample.task import build_model, evaluate, train


def floating_state(model):
    """Return floating tensors in their stable state_dict order."""
    return {
        name: tensor
        for name, tensor in model.model.state_dict().items()
        if tensor.is_floating_point()
    }


class JetsonYoloClient(fl.client.NumPyClient):
    """Classic Flower client compatible with Python 3.8 and Flower 1.10."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        dataset_config = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
        names = dataset_config["names"]
        class_names = (
            {int(index): name for index, name in names.items()}
            if isinstance(names, dict)
            else dict(enumerate(names))
        )
        self.model = build_model(args.model, class_names=class_names)
        self.parameter_names = list(floating_state(self.model))

    def release_resources(self, phase: str) -> None:
        """Release Ultralytics runtime objects and CUDA cache between phases."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            print(
                f"Node {self.args.node_id}: before {phase} cleanup: "
                f"CUDA allocated={allocated:.1f} MiB, reserved={reserved:.1f} MiB"
            )

        # Keep only model weights. Trainers, validators and predictors can retain
        # dataloaders, batches, plots and CUDA tensors after a phase has ended.
        self.model.model.to("cpu")
        self.model.trainer = None
        self.model.validator = None
        self.model.predictor = None
        self.model.metrics = None
        gc.collect()
        try:
            # Return free glibc heap pages to Linux instead of retaining them in
            # this long-running Flower process between federated rounds.
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (AttributeError, OSError):
            pass

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            print(
                f"Node {self.args.node_id}: after {phase} cleanup: "
                f"CUDA allocated={torch.cuda.memory_allocated() / (1024**2):.1f} MiB, "
                f"reserved={torch.cuda.memory_reserved() / (1024**2):.1f} MiB"
            )

    def get_parameters(self, config: Mapping):
        state = floating_state(self.model)
        return [state[name].detach().cpu().numpy().copy() for name in self.parameter_names]

    def set_parameters(self, parameters) -> None:
        if len(parameters) != len(self.parameter_names):
            raise ValueError(
                f"Expected {len(self.parameter_names)} tensors, got {len(parameters)}"
            )
        local_state = self.model.model.state_dict()
        for name, array in zip(self.parameter_names, parameters):
            target = local_state[name]
            local_state[name] = torch.from_numpy(np.asarray(array)).to(
                device=target.device, dtype=target.dtype
            )
        self.model.model.load_state_dict(local_state, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", self.args.local_epochs))
        print(
            f"Node {self.args.node_id}: training {self.args.data} "
            f"for {epochs} epoch(s)"
        )
        try:
            train(
                model=self.model,
                data=self.args.data,
                epochs=epochs,
                image_size=self.args.image_size,
                batch_size=self.args.batch_size,
                device=self.args.device,
                learning_rate=self.args.learning_rate,
                project=f"/app/runs/client_{self.args.node_id}",
                workers=self.args.workers,
            )
            num_examples = len(self.model.trainer.train_loader.dataset)
        except BaseException:
            self.release_resources("failed train")
            raise
        self.release_resources("train")
        return self.get_parameters({}), num_examples, {"node_id": self.args.node_id}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        print(f"Node {self.args.node_id}: validating global model on {self.args.data}")
        try:
            results = evaluate(
                model=self.model,
                data=self.args.data,
                image_size=self.args.image_size,
                batch_size=self.args.batch_size,
                device=self.args.device,
                workers=self.args.workers,
            )
            num_examples = max(int(results.nt_per_class.sum()), 1)
            metrics = {
                "node_id": self.args.node_id,
                "map50": float(results.box.map50),
                "map50-95": float(results.box.map),
                "precision": float(results.box.mp),
                "recall": float(results.box.mr),
                "num-examples": num_examples,
            }
        except BaseException:
            self.release_resources("failed validation")
            raise
        self.release_resources("validation")
        return 1.0 - metrics["map50-95"], num_examples, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, help="Flower server, e.g. 192.168.1.10:8080")
    parser.add_argument("--node-id", required=True, help="Stable node name/ID for logs")
    parser.add_argument("--data", required=True, help="Absolute path to this client's data.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers; keep 0 on Jetson Nano to avoid shared-memory errors",
    )
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Torch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available inside the container")
    fl.client.start_client(
        server_address=args.server,
        client=JetsonYoloClient(args).to_client(),
    )


if __name__ == "__main__":
    main()
