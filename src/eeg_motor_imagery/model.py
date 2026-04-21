from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from tensorflow import keras


def set_reproducibility(seed: int) -> None:
    """Set the main random seeds used by NumPy, Python, and TensorFlow."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
    except Exception:
        # TensorFlow seeding can vary slightly across backends and versions.
        pass


def build_eegnet(
    input_shape: tuple[int, int, int],
    n_classes: int,
    p_drop: float = 0.5,
    temporal_kernel_size: tuple[int, int] = (1, 65),
    n_temporal_kernels: int = 8,
    spatial_depth_multiplier: int = 2,
    separable_temporal_kernel_size: tuple[int, int] = (1, 17),
) -> keras.Model:
    """Build EEGNet with the same topology used in the original exercise."""

    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Conv2D(
                n_temporal_kernels,
                kernel_size=temporal_kernel_size,
                padding="same",
                use_bias=False,
            ),
            keras.layers.BatchNormalization(),
            keras.layers.DepthwiseConv2D(
                kernel_size=(input_shape[0], 1),
                use_bias=False,
                depth_multiplier=spatial_depth_multiplier,
                depthwise_constraint=keras.constraints.max_norm(1.0),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("elu"),
            keras.layers.AveragePooling2D(pool_size=(1, 4)),
            keras.layers.Dropout(p_drop),
            keras.layers.SeparableConv2D(
                n_temporal_kernels * spatial_depth_multiplier,
                kernel_size=separable_temporal_kernel_size,
                use_bias=False,
                padding="same",
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("elu"),
            keras.layers.AveragePooling2D(pool_size=(1, 8)),
            keras.layers.Dropout(p_drop),
            keras.layers.Flatten(),
            keras.layers.Dense(
                n_classes,
                kernel_constraint=keras.constraints.max_norm(0.25),
                activation="softmax",
            ),
        ]
    )


def save_model_summary(model: keras.Model, destination: str | Path) -> None:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(f"{line}\n"))
