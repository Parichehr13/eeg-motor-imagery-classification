from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from .data import DatasetBundle


def _prepare_output(destination: str | Path) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_label_histograms(dataset: DatasetBundle, destination: str | Path) -> None:
    output_path = _prepare_output(destination)
    splits = [("train", dataset.train.labels), ("valid", dataset.valid.labels), ("test", dataset.test.labels)]
    class_indices = np.arange(len(dataset.class_names))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, (split_name, labels) in zip(axes, splits):
        counts = [int(np.sum(labels == class_index)) for class_index in class_indices]
        axis.bar(class_indices, counts, color="#9aa4b2", edgecolor="black")
        axis.set_xticks(class_indices, dataset.class_names, rotation=30, ha="right")
        axis.set_title(split_name.capitalize())
        axis.set_xlabel("Class")
        axis.set_ylabel("Trials")
    fig.suptitle("Class distribution across splits")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: dict[str, list[float]], destination: str | Path) -> None:
    output_path = _prepare_output(destination)
    epochs = np.arange(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(epochs, history["loss"], label="train", color="#1f2933")
    axes[0].plot(epochs, history["val_loss"], label="valid", color="#d1495b")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["accuracy"], label="train", color="#1f2933")
    axes[1].plot(epochs, history["val_accuracy"], label="valid", color="#d1495b")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle("Training history")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(metrics: dict, class_names: list[str], destination: str | Path) -> None:
    output_path = _prepare_output(destination)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for axis, split_name in zip(axes, ("train", "valid", "test")):
        matrix = np.array(metrics[split_name]["confusion_matrix"])
        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
        display.plot(ax=axis, colorbar=False, cmap="Blues")
        axis.set_title(split_name.capitalize())
        axis.tick_params(axis="x", rotation=30)

    fig.suptitle("Confusion matrices")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_importance(channel_rows: list[dict], destination: str | Path) -> None:
    output_path = _prepare_output(destination)
    ordered = sorted(channel_rows, key=lambda row: row["importance"], reverse=True)
    labels = [row["channel"] for row in ordered]
    scores = [row["importance"] for row in ordered]
    errors = [row["importance_std"] for row in ordered]
    colors = ["#1b4332" if index < 5 else "#74a57f" for index in range(len(labels))]

    fig, axis = plt.subplots(figsize=(13, 5.5))
    axis.bar(
        np.arange(len(labels)),
        scores,
        yerr=errors,
        capsize=3,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )
    axis.set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right")
    axis.set_ylabel("Average absolute spatial weight")
    axis.set_xlabel("EEG channel")
    axis.set_title("Ranked channel importance from EEGNet spatial filters")
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
