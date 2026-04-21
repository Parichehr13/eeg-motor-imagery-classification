from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow import keras

from .data import DatasetBundle, SplitData


def _to_serializable(value: Any):
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def save_json(data: dict, destination: str | Path) -> None:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_to_serializable(data), indent=2),
        encoding="utf-8",
    )


def evaluate_split(split: SplitData, probabilities: np.ndarray, class_names: list[str]) -> dict:
    y_true = split.labels
    y_pred = np.argmax(probabilities, axis=1)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "predicted_labels": y_pred.tolist(),
    }


def evaluate_model(model: keras.Model, dataset: DatasetBundle) -> dict:
    metrics = {}
    for split_name in ("train", "valid", "test"):
        split = getattr(dataset, split_name)
        probabilities = model.predict(split.x, verbose=0)
        metrics[split_name] = evaluate_split(split, probabilities, dataset.class_names)
    return metrics


def extract_spatial_channel_importance(model: keras.Model, channel_names: list[str]) -> list[dict]:
    spatial_layer = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.DepthwiseConv2D):
            spatial_layer = layer
            break

    if spatial_layer is None:
        raise ValueError("No depthwise spatial convolution layer found in the model.")

    weights = spatial_layer.get_weights()[0]
    abs_weights = np.abs(weights).reshape((weights.shape[0], -1))
    importance = abs_weights.mean(axis=1)
    importance_std = abs_weights.std(axis=1)
    normalized = importance / np.sum(importance)
    ranking = np.argsort(importance)[::-1]
    return [
        {
            "rank": int(rank + 1),
            "channel": channel_names[channel_index],
            "importance": float(importance[channel_index]),
            "importance_std": float(importance_std[channel_index]),
            "normalized_importance": float(normalized[channel_index]),
        }
        for rank, channel_index in enumerate(ranking)
    ]


def save_channel_importance(rows: list[dict], destination: str | Path) -> None:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "channel",
                "importance",
                "importance_std",
                "normalized_importance",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
