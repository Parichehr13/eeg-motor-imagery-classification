from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class RunConfig:
    data_path: str = "data/raw/bci_iv2a_sub-008.mat"
    model_dir: str = "models"
    results_dir: str = "results"
    validation_ratio: float = 0.1
    validation_strategy: str = "stratified_shuffle"
    optimizer_name: str = "sgd"
    learning_rate: float = 0.001
    momentum: float = 0.9
    batch_size: int = 64
    max_epochs: int = 1000
    random_seed: int = 42
    early_stopping_patience: int = 30
    reduce_lr_patience: int = 12
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-5

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def figures_dir(self) -> Path:
        return Path(self.results_dir) / "figures"

    @property
    def metrics_dir(self) -> Path:
        return Path(self.results_dir) / "metrics"

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.model_dir) / "eegnet_best.weights.h5"


def load_config(path: str | Path | None = None) -> RunConfig:
    config = RunConfig()
    if path is None:
        return config

    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    values = config.to_dict()
    values.update(data)
    return RunConfig(**values)


def save_config(config: RunConfig, destination: str | Path) -> None:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config.to_dict(), indent=2),
        encoding="utf-8",
    )
