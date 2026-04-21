from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.io import savemat


def make_test_dir(root: str | Path, prefix: str) -> Path:
    path = Path(root) / ".test_runs" / f"{prefix}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_synthetic_bci_iv2a_mat(
    destination: str | Path,
    *,
    n_train_per_class: int = 4,
    n_test_per_class: int = 3,
    n_channels: int = 6,
    n_timepoints: int = 32,
    seed: int = 7,
) -> Path:
    rng = np.random.default_rng(seed)
    class_names = ["left_hand", "right_hand", "feet", "tongue"]
    channel_names = [f"C{index + 1}" for index in range(n_channels)]

    trials = []
    labels = []
    sessions = []

    time_axis = np.linspace(0.0, 1.0, n_timepoints, endpoint=False)
    base_temporal_patterns = [
        np.sin(2.0 * np.pi * 6.0 * time_axis),
        np.cos(2.0 * np.pi * 8.0 * time_axis),
        np.sin(2.0 * np.pi * 10.0 * time_axis + 0.4),
        np.cos(2.0 * np.pi * 12.0 * time_axis + 0.8),
    ]

    for class_index, pattern in enumerate(base_temporal_patterns):
        spatial_weights = np.linspace(1.0, 2.0, n_channels, dtype=np.float32)
        spatial_weights = np.roll(spatial_weights, class_index)
        template = np.outer(spatial_weights, pattern).astype(np.float32)

        for _ in range(n_train_per_class):
            trials.append(template + 0.15 * rng.standard_normal((n_channels, n_timepoints), dtype=np.float32))
            labels.append(class_index + 1)
            sessions.append("session_T")

        for _ in range(n_test_per_class):
            trials.append(template + 0.15 * rng.standard_normal((n_channels, n_timepoints), dtype=np.float32))
            labels.append(class_index + 1)
            sessions.append("session_E")

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        output_path,
        {
            "x": np.stack(trials).astype(np.float32),
            "y": np.asarray(labels, dtype=np.int64).reshape(-1, 1),
            "sf": np.asarray([[128]], dtype=np.int64),
            "channels": np.asarray([[name] for name in channel_names], dtype=object),
            "events": np.asarray([[name] for name in class_names], dtype=object),
            "session": np.asarray(sessions, dtype=object).reshape(-1, 1),
        },
    )
    return output_path
