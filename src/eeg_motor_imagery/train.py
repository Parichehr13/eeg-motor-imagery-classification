from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tensorflow import keras

from .config import RunConfig, load_config, save_config
from .data import prepare_dataset
from .evaluation import evaluate_model, extract_spatial_channel_importance, save_channel_importance, save_json
from .model import build_eegnet, save_model_summary, set_reproducibility
from .plotting import plot_confusion_matrices, plot_label_histograms, plot_spatial_importance, plot_training_curves


def _write_history_csv(history: dict[str, list[float]], destination: str | Path) -> None:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(history.keys())
    n_rows = len(history[columns[0]])
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *columns])
        for index in range(n_rows):
            row = [index + 1]
            for column in columns:
                row.append(history[column][index])
            writer.writerow(row)


def _write_results_summary(
    config: RunConfig,
    dataset,
    metrics: dict,
    channel_importance: list[dict],
    history: dict[str, list[float]],
    destination: str | Path,
) -> None:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_channels = ", ".join(
        f"{row['channel']} ({row['importance']:.4f})" for row in channel_importance[:5]
    )
    test_metrics = metrics["test"]
    best_epoch = min(range(len(history["val_loss"])), key=lambda index: history["val_loss"][index]) + 1
    output_path.write_text(
        "\n".join(
            [
                "# Run Summary",
                "",
                f"- Validation strategy: `{config.validation_strategy}`",
                f"- Optimizer: `{config.optimizer_name}`",
                f"- Dataset file: `{config.data_path}`",
                f"- Sampling rate: {dataset.sampling_rate} Hz",
                f"- Train/valid/test trials: {dataset.train.x.shape[0]}/{dataset.valid.x.shape[0]}/{dataset.test.x.shape[0]}",
                f"- Train split counts: {json.dumps(dataset.split_summary['train'])}",
                f"- Valid split counts: {json.dumps(dataset.split_summary['valid'])}",
                f"- Standardization mean/std (train only): {dataset.train_mean:.6f} / {dataset.train_std:.6f}",
                f"- Best epoch by validation loss: {best_epoch}",
                f"- Best validation loss: {history['val_loss'][best_epoch - 1]:.4f}",
                f"- Best validation accuracy: {history['val_accuracy'][best_epoch - 1]:.4f}",
                f"- Test accuracy: {test_metrics['accuracy']:.4f}",
                f"- Test macro F1: {test_metrics['f1_macro']:.4f}",
                f"- Top spatially weighted channels: {top_channels}",
                "",
                "Interpretation note:",
                "The spatial-filter ranking is a lightweight interpretability view of EEGNet's depthwise spatial layer.",
                "It shows which electrodes received larger absolute weights in this single-subject training run.",
                "These scores are useful for inspection, but they should not be treated as a causal neuroscientific finding.",
            ]
        ),
        encoding="utf-8",
    )


def _build_optimizer(config: RunConfig):
    if config.optimizer_name == "sgd":
        return keras.optimizers.SGD(
            learning_rate=config.learning_rate,
            momentum=config.momentum,
        )
    if config.optimizer_name == "adam":
        return keras.optimizers.Adam(learning_rate=config.learning_rate)
    raise ValueError("optimizer_name must be either 'sgd' or 'adam'.")


def _build_callbacks(config: RunConfig) -> list[keras.callbacks.Callback]:
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )
    ]

    if config.early_stopping_patience > 0:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
                restore_best_weights=False,
            )
        )
    if config.reduce_lr_patience > 0:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=config.reduce_lr_factor,
                patience=config.reduce_lr_patience,
                min_lr=config.min_learning_rate,
                verbose=1,
            )
        )
    return callbacks


def run_experiment(config: RunConfig) -> dict:
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.metrics_dir.mkdir(parents=True, exist_ok=True)

    set_reproducibility(config.random_seed)
    dataset = prepare_dataset(
        config.data_path,
        validation_ratio=config.validation_ratio,
        random_seed=config.random_seed,
        validation_strategy=config.validation_strategy,
    )

    plot_label_histograms(dataset, config.figures_dir / "class_histograms.png")

    model = build_eegnet(dataset.input_shape, dataset.n_classes)
    save_model_summary(model, config.metrics_dir / "model_summary.txt")

    optimizer = _build_optimizer(config)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    history = model.fit(
        dataset.train.x,
        dataset.train.y,
        batch_size=config.batch_size,
        epochs=config.max_epochs,
        validation_data=(dataset.valid.x, dataset.valid.y),
        callbacks=_build_callbacks(config),
        verbose=2,
    ).history

    history = {key: [float(value) for value in values] for key, values in history.items()}
    _write_history_csv(history, config.metrics_dir / "training_history.csv")
    save_json(history, config.metrics_dir / "training_history.json")
    plot_training_curves(history, config.figures_dir / "training_curves.png")

    best_model = build_eegnet(dataset.input_shape, dataset.n_classes)
    best_model.load_weights(config.checkpoint_path)
    metrics = evaluate_model(best_model, dataset)
    save_json(metrics, config.metrics_dir / "metrics.json")
    plot_confusion_matrices(metrics, dataset.class_names, config.figures_dir / "confusion_matrices.png")

    channel_importance = extract_spatial_channel_importance(best_model, dataset.channel_names)
    save_channel_importance(channel_importance, config.metrics_dir / "channel_importance.csv")
    plot_spatial_importance(channel_importance, config.figures_dir / "spatial_filter_importance.png")

    save_config(config, config.metrics_dir / "run_config.json")
    _write_results_summary(
        config,
        dataset,
        metrics,
        channel_importance,
        history,
        config.metrics_dir / "results_summary.md",
    )

    return {
        "config": config.to_dict(),
        "metrics": metrics,
        "channel_importance": channel_importance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEGNet on the provided BCI IV 2a subject.")
    parser.add_argument(
        "--config",
        default="configs/default_run.json",
        help="Path to the JSON config file.",
    )
    parser.add_argument("--data-path", default=None, help="Override the dataset path from the config.")
    parser.add_argument("--results-dir", default=None, help="Override the results directory from the config.")
    parser.add_argument("--model-dir", default=None, help="Override the model directory from the config.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the training batch size.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override the maximum number of epochs.")
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the optimizer learning rate.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=None,
        help="Override SGD momentum.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=None,
        help="Override the validation ratio.",
    )
    parser.add_argument(
        "--validation-strategy",
        default=None,
        help="Override the validation strategy: first_block or stratified_shuffle.",
    )
    parser.add_argument(
        "--optimizer",
        default=None,
        help="Override the optimizer: sgd or adam.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Override early stopping patience. Use 0 to disable.",
    )
    parser.add_argument(
        "--reduce-lr-patience",
        type=int,
        default=None,
        help="Override ReduceLROnPlateau patience. Use 0 to disable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    if args.data_path is not None:
        config.data_path = args.data_path
    if args.results_dir is not None:
        config.results_dir = args.results_dir
    if args.model_dir is not None:
        config.model_dir = args.model_dir
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.seed is not None:
        config.random_seed = args.seed
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.momentum is not None:
        config.momentum = args.momentum
    if args.validation_ratio is not None:
        config.validation_ratio = args.validation_ratio
    if args.validation_strategy is not None:
        config.validation_strategy = args.validation_strategy
    if args.optimizer is not None:
        config.optimizer_name = args.optimizer
    if args.early_stopping_patience is not None:
        config.early_stopping_patience = args.early_stopping_patience
    if args.reduce_lr_patience is not None:
        config.reduce_lr_patience = args.reduce_lr_patience

    run_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
