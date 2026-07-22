"""Flower classic server compatible with JetPack 4/Flower 1.10 clients."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import flwr as fl
import torch
import yaml
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from embeddedexample.task import (
    build_model,
    get_trainable_state,
    parse_merge_parts,
    set_trainable_state,
)


METRIC_NAMES = ("map50", "map50-95", "precision", "recall", "num-examples")


def weighted_metrics(results):
    total = sum(num_examples for num_examples, _ in results)
    if total == 0:
        return {}
    keys = ("map50", "map50-95", "precision", "recall")
    return {
        key: sum(num_examples * float(metrics[key]) for num_examples, metrics in results)
        / total
        for key in keys
    }


class CheckpointingFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self, *, model, merge_parts, output_path: Path, metrics_path: Path, **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.output_path = output_path
        self.metrics_path = metrics_path
        self.merge_parts = merge_parts

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is not None:
            arrays = parameters_to_ndarrays(parameters)
            names = list(get_trainable_state(self.model, self.merge_parts))
            if len(arrays) != len(names):
                raise ValueError(
                    f"Expected {len(names)} aggregated tensors, got {len(arrays)}"
                )
            state = {
                name: torch.from_numpy(array) for name, array in zip(names, arrays)
            }
            set_trainable_state(self.model, state)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.output_path))
            print(f"Saved round {server_round} global model to {self.output_path}")
        return parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        if results:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = (
                not self.metrics_path.exists() or self.metrics_path.stat().st_size == 0
            )
            fields = ("round", "node_id", *METRIC_NAMES)
            with self.metrics_path.open("a", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                if write_header:
                    writer.writeheader()
                for client, evaluate_result in results:
                    metrics = evaluate_result.metrics
                    writer.writerow(
                        {
                            "round": server_round,
                            "node_id": metrics.get("node_id", client.cid),
                            **{name: metrics.get(name, "") for name in METRIC_NAMES},
                        }
                    )
        return super().aggregate_evaluate(server_round, results, failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="0.0.0.0:8080")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument(
        "--data-config",
        type=Path,
        required=True,
        help="YOLO data.yaml used to configure the global model class head",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--config", type=Path, default=Path("/app/config.yaml"))
    parser.add_argument(
        "--merge-parts",
        default=None,
        help="Override config with comma-separated parts, e.g. backbone,head",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/fl/final_yolo11.pt"))
    parser.add_argument(
        "--metrics", type=Path, default=Path("runs/fl/val_metrics_by_node.csv")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configured_parts = None
    if args.config.is_file():
        config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        configured_parts = config.get("training", {}).get("merge_parts")
    merge_parts = parse_merge_parts(
        args.merge_parts or configured_parts or ("backbone", "neck", "head")
    )
    dataset_config = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))
    names = dataset_config["names"]
    class_names = (
        {int(index): name for index, name in names.items()}
        if isinstance(names, dict)
        else dict(enumerate(names))
    )
    model = build_model(args.model, class_names=class_names)
    print(f"Initialized global model with {len(class_names)} classes: {class_names}")
    initial_arrays = [
        tensor.numpy() for tensor in get_trainable_state(model, merge_parts).values()
    ]
    print(f"Federated model parts: {', '.join(merge_parts)}")
    strategy = CheckpointingFedAvg(
        model=model,
        merge_parts=merge_parts,
        output_path=args.output,
        metrics_path=args.metrics,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.clients,
        min_evaluate_clients=args.clients,
        min_available_clients=args.clients,
        initial_parameters=ndarrays_to_parameters(initial_arrays),
        on_fit_config_fn=lambda _round: {
            "local_epochs": args.local_epochs,
            "merge_parts": ",".join(merge_parts),
        },
        on_evaluate_config_fn=lambda _round: {
            "merge_parts": ",".join(merge_parts)
        },
        evaluate_metrics_aggregation_fn=weighted_metrics,
    )
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
