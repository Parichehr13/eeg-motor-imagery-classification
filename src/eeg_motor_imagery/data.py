from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


@dataclass(slots=True)
class SplitData:
    x: np.ndarray
    y: np.ndarray
    labels: np.ndarray


@dataclass(slots=True)
class DatasetBundle:
    train: SplitData
    valid: SplitData
    test: SplitData
    sampling_rate: int
    channel_names: list[str]
    class_names: list[str]
    input_shape: tuple[int, int, int]
    n_classes: int
    train_mean: float
    train_std: float
    split_summary: dict[str, dict[str, int]]


def load_bci_iv2a(filepath: str | Path):
    """Load the provided BCI Competition IV 2a subject file."""

    mat = loadmat(filepath)
    x = mat["x"]
    y = np.squeeze(mat["y"]).astype(np.int64) - 1
    channel_names = [channel[0] for channel in np.squeeze(mat["channels"])]
    sampling_rate = int(np.squeeze(mat["sf"]))
    session = np.squeeze(mat["session"])
    class_names = [event[0] for event in np.squeeze(mat["events"])]

    train_idx = np.where(session == "session_T")[0]
    test_idx = np.where(session == "session_E")[0]

    x_train = x[train_idx, :, :].astype(np.float32)
    x_test = x[test_idx, :, :].astype(np.float32)
    y_train = y[train_idx]
    y_test = y[test_idx]
    return (x_train, y_train), (x_test, y_test), sampling_rate, channel_names, class_names


def to_one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert integer labels into one-hot encoded labels."""

    labels = labels.astype(np.int64)
    one_hot = np.zeros((labels.shape[0], n_classes), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


def _label_distribution(labels: np.ndarray, class_names: list[str]) -> dict[str, int]:
    return {
        class_name: int(np.sum(labels == class_index))
        for class_index, class_name in enumerate(class_names)
    }


def prepare_dataset(
    filepath: str | Path,
    validation_ratio: float = 0.1,
    random_seed: int = 42,
    validation_strategy: str = "stratified_shuffle",
) -> DatasetBundle:
    (x_train_full, labels_train_full), (x_test, labels_test), sampling_rate, channel_names, class_names = (
        load_bci_iv2a(filepath)
    )

    n_channels = x_train_full.shape[1]
    n_timepoints = x_train_full.shape[2]
    n_classes = len(class_names)

    x_train_full = x_train_full[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    y_train_full = to_one_hot(labels_train_full, n_classes)
    y_test = to_one_hot(labels_test, n_classes)

    if validation_strategy == "first_block":
        valid_size = max(1, int(round(validation_ratio * x_train_full.shape[0])))
        x_valid = x_train_full[:valid_size]
        y_valid = y_train_full[:valid_size]
        labels_valid = labels_train_full[:valid_size]

        x_train = x_train_full[valid_size:]
        y_train = y_train_full[valid_size:]
        labels_train = labels_train_full[valid_size:]
    elif validation_strategy == "stratified_shuffle":
        x_train, x_valid, y_train, y_valid, labels_train, labels_valid = train_test_split(
            x_train_full,
            y_train_full,
            labels_train_full,
            test_size=validation_ratio,
            random_state=random_seed,
            stratify=labels_train_full,
            shuffle=True,
        )
    else:
        raise ValueError(
            "validation_strategy must be either 'first_block' or 'stratified_shuffle'."
        )

    train_mean = float(np.mean(x_train))
    train_std = float(np.std(x_train))
    scale = train_std + 1e-15

    x_train = (x_train - train_mean) / scale
    x_valid = (x_valid - train_mean) / scale
    x_test = (x_test - train_mean) / scale

    return DatasetBundle(
        train=SplitData(x=x_train, y=y_train, labels=labels_train),
        valid=SplitData(x=x_valid, y=y_valid, labels=labels_valid),
        test=SplitData(x=x_test, y=y_test, labels=labels_test),
        sampling_rate=sampling_rate,
        channel_names=channel_names,
        class_names=class_names,
        input_shape=(n_channels, n_timepoints, 1),
        n_classes=n_classes,
        train_mean=train_mean,
        train_std=train_std,
        split_summary={
            "train": _label_distribution(labels_train, class_names),
            "valid": _label_distribution(labels_valid, class_names),
            "test": _label_distribution(labels_test, class_names),
        },
    )
