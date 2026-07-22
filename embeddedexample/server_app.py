"""Flower ServerApp for federated YOLO11 training."""

import csv
from collections.abc import Iterable
from pathlib import Path

from flwr.app import ArrayRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from embeddedexample.task import (
    build_model,
    get_trainable_state,
    load_class_names,
    parse_merge_parts,
    set_trainable_state,
)

app = ServerApp()


class MetricsLoggingFedAvg(FedAvg):
    """FedAvg strategy which persists every client's global-model val metrics."""

    _METRIC_NAMES = ("map50", "map50-95", "precision", "recall", "num-examples")

    def __init__(self, *, metrics_path: str, run_id: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.metrics_path = Path(metrics_path)
        self.run_id = run_id

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> MetricRecord | None:
        """Log each valid reply, then perform the normal weighted aggregation."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)
        if not valid_replies:
            return None

        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not self.metrics_path.exists() or self.metrics_path.stat().st_size == 0
        )
        with self.metrics_path.open("a", encoding="utf-8", newline="") as csv_file:
            fieldnames = ("run_id", "round", "node_id", *self._METRIC_NAMES)
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for reply in valid_replies:
                metrics = reply.content["metrics"]
                writer.writerow(
                    {
                        "run_id": self.run_id,
                        "round": server_round,
                        "node_id": reply.metadata.src_node_id,
                        **{name: metrics.get(name, "") for name in self._METRIC_NAMES},
                    }
                )

        print(
            f"Saved {len(valid_replies)} client validation result(s) "
            f"for round {server_round} to {self.metrics_path.resolve()}"
        )
        reply_contents = [reply.content for reply in valid_replies]
        return self.evaluate_metrics_aggr_fn(reply_contents, self.weighted_by_key)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run FedAvg and save a directly usable Ultralytics checkpoint."""
    pretrained_model = str(context.run_config["pretrained-model"])
    merge_parts = parse_merge_parts(context.run_config["merge-parts"])
    print(f"Loading initial global model from: {pretrained_model}")
    print(f"Federated model parts: {', '.join(merge_parts)}")
    dataset_config = str(context.run_config["dataset-config"]).format(partition_id=0)
    global_model = build_model(
        pretrained_model, class_names=load_class_names(dataset_config)
    )
    strategy = MetricsLoggingFedAvg(
        metrics_path=str(context.run_config["val-metrics-path"]),
        run_id=context.run_id,
        fraction_train=float(context.run_config["fraction-train"]),
        fraction_evaluate=float(context.run_config["fraction-evaluate"]),
        min_train_nodes=int(context.run_config["min-train-nodes"]),
        min_evaluate_nodes=int(context.run_config["min-evaluate-nodes"]),
        min_available_nodes=int(context.run_config["min-available-nodes"]),
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(get_trainable_state(global_model, merge_parts)),
        num_rounds=int(context.run_config["num-server-rounds"]),
    )

    set_trainable_state(global_model, result.arrays.to_torch_state_dict())
    output = Path(str(context.run_config["output-path"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    global_model.save(str(output))
    print(f"\nSaved federated YOLO11 model to {output.resolve()}")
