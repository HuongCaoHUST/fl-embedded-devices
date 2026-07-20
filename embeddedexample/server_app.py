"""Flower ServerApp for federated YOLO11 training."""

from pathlib import Path

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from embeddedexample.task import build_model, get_trainable_state, set_trainable_state

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run FedAvg and save a directly usable Ultralytics checkpoint."""
    model_name = str(context.run_config["model-name"])
    global_model = build_model(model_name)
    strategy = FedAvg(
        fraction_train=float(context.run_config["fraction-train"]),
        fraction_evaluate=float(context.run_config["fraction-evaluate"]),
        min_train_nodes=int(context.run_config["min-train-nodes"]),
        min_evaluate_nodes=int(context.run_config["min-evaluate-nodes"]),
        min_available_nodes=int(context.run_config["min-available-nodes"]),
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(get_trainable_state(global_model)),
        num_rounds=int(context.run_config["num-server-rounds"]),
    )

    set_trainable_state(global_model, result.arrays.to_torch_state_dict())
    output = Path(str(context.run_config["output-path"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    global_model.save(str(output))
    print(f"\nSaved federated YOLO11 model to {output.resolve()}")
