"""Submit one Flower run automatically according to config.yaml."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import yaml


CONFIG_PATH = Path("/app/config.yaml")


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    training = config.get("training", {})
    if not training.get("auto_start", False):
        print("Automatic training is disabled in config.yaml")
        return

    minimum_clients = int(training.get("minimum_clients", 2))
    local_epochs = int(training.get("local_epochs", 1))
    server_rounds = int(training.get("server_rounds", 2))
    startup_delay = float(training.get("startup_delay_seconds", 5))
    if minimum_clients < 1:
        raise ValueError("training.minimum_clients must be at least 1")
    if startup_delay < 0:
        raise ValueError("training.startup_delay_seconds cannot be negative")
    if local_epochs < 1:
        raise ValueError("training.local_epochs must be at least 1")
    if server_rounds < 1:
        raise ValueError("training.server_rounds must be at least 1")

    print(
        f"Auto-start enabled: waiting {startup_delay:g}s before submitting; "
        f"Flower will wait for at least {minimum_clients} client(s), then run "
        f"{server_rounds} round(s) with {local_epochs} local epoch(s) per round."
    )
    time.sleep(startup_delay)
    run_config = " ".join(
        [
            f"min-available-nodes={minimum_clients}",
            f"min-train-nodes={minimum_clients}",
            f"min-evaluate-nodes={minimum_clients}",
            f"local-epochs={local_epochs}",
            f"num-server-rounds={server_rounds}",
        ]
    )
    subprocess.run(
        ["flwr", "run", ".", "docker", "--stream", "--run-config", run_config],
        check=True,
    )


if __name__ == "__main__":
    main()
