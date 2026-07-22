"""Flower ClientApp for federated YOLO11 object detection."""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from embeddedexample.task import (
    build_model,
    evaluate as evaluate_model,
    get_trainable_state,
    load_class_names,
    parse_merge_parts,
    resolve_dataset_config,
    set_trainable_state,
    train as train_model,
)

app = ClientApp()


def _device(context: Context) -> str:
    requested = str(context.run_config.get("device", "auto"))
    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return requested


def _load_global_model(msg: Message, context: Context):
    # Instantiate the exact same architecture/checkpoint as the server, then
    # overwrite its floating tensors with the current federated global state.
    dataset_config = resolve_dataset_config(context.node_config, context.run_config)
    model = build_model(
        str(context.run_config["pretrained-model"]),
        class_names=load_class_names(dataset_config),
    )
    merge_parts = parse_merge_parts(context.run_config["merge-parts"])
    partition_id = int(context.node_config.get("partition-id", 0))
    if "local-arrays" in context.state:
        set_trainable_state(
            model, context.state["local-arrays"].to_torch_state_dict()
        )
    global_state = msg.content["arrays"].to_torch_state_dict()
    expected_names = set(get_trainable_state(model, merge_parts))
    if set(global_state) != expected_names:
        raise ValueError(
            f"Global tensor selection mismatch for client {partition_id}: "
            f"expected {len(expected_names)}, received {len(global_state)}"
        )
    set_trainable_state(model, global_state)
    full_tensor_count = len(get_trainable_state(model))
    print(
        f"Client partition {partition_id} loaded {len(global_state)}/"
        f"{full_tensor_count} global tensors for {','.join(merge_parts)}; "
        f"kept {full_tensor_count - len(global_state)} tensors local"
    )
    return model


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Load global weights, train on this client's images, and return new weights."""
    model = _load_global_model(msg, context)
    data = resolve_dataset_config(context.node_config, context.run_config)
    partition_id = int(context.node_config.get("partition-id", 0))
    print(f"Client partition {partition_id} training with dataset: {data}")
    train_model(
        model=model,
        data=data,
        epochs=int(context.run_config["local-epochs"]),
        image_size=int(context.run_config["image-size"]),
        batch_size=int(context.run_config["batch-size"]),
        device=_device(context),
        learning_rate=float(context.run_config["learning-rate"]),
        project=f"runs/fl/client_{partition_id}",
    )
    num_examples = len(model.trainer.train_loader.dataset)
    loss_parts = [
        float(value)
        for key, value in model.trainer.metrics.items()
        if key.startswith("train/") and key.endswith("_loss")
    ]
    loss = sum(loss_parts)
    # Preserve every local tensor across rounds. Only selected tensors are sent
    # to the server and overwritten by the next global model.
    context.state["local-arrays"] = ArrayRecord(get_trainable_state(model))
    merge_parts = parse_merge_parts(context.run_config["merge-parts"])
    content = RecordDict(
        {
            "arrays": ArrayRecord(get_trainable_state(model, merge_parts)),
            "metrics": MetricRecord(
                {"train_loss": loss, "num-examples": num_examples}
            ),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate global weights on this client's validation split."""
    model = _load_global_model(msg, context)
    data = resolve_dataset_config(context.node_config, context.run_config)
    partition_id = int(context.node_config.get("partition-id", 0))
    print(f"Client partition {partition_id} validating with dataset: {data}")
    metrics = evaluate_model(
        model=model,
        data=data,
        image_size=int(context.run_config["image-size"]),
        batch_size=int(context.run_config["batch-size"]),
        device=_device(context),
    )
    # Ultralytics exposes target counts in DetMetrics. Weighting by the number
    # of annotated objects avoids giving a tiny client the same influence as a
    # much larger one during distributed evaluation.
    num_examples = max(int(metrics.nt_per_class.sum()), 1)
    values = {
        "map50": float(metrics.box.map50),
        "map50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "num-examples": num_examples,
    }
    return Message(
        content=RecordDict({"metrics": MetricRecord(values)}), reply_to=msg
    )
