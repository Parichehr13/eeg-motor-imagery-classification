from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .config import RunConfig, load_config
from .data import DatasetBundle, prepare_dataset
from .evaluation import evaluate_probabilities_by_split, save_json


def _squeeze_trials(x: np.ndarray) -> np.ndarray:
    if x.ndim == 4 and x.shape[-1] == 1:
        return x[..., 0]
    return x


def _mean_normalized_covariance(trials: np.ndarray) -> np.ndarray:
    covariances = []
    for trial in trials:
        covariance = trial @ trial.T
        covariance = covariance / (np.trace(covariance) + 1e-12)
        covariances.append(covariance)
    return np.mean(covariances, axis=0)


def _fit_binary_csp_filters(
    x_train: np.ndarray,
    positive_mask: np.ndarray,
    n_components: int,
) -> np.ndarray:
    if n_components < 2:
        raise ValueError("baseline_csp_components must be at least 2.")
    if n_components % 2 != 0:
        raise ValueError("baseline_csp_components must be even for symmetric CSP filters.")

    class_covariance = _mean_normalized_covariance(x_train[positive_mask])
    rest_covariance = _mean_normalized_covariance(x_train[~positive_mask])
    composite_covariance = class_covariance + rest_covariance

    eigenvalues, eigenvectors = np.linalg.eigh(composite_covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    whitening = np.diag(1.0 / np.sqrt(np.clip(eigenvalues, 1e-12, None))) @ eigenvectors.T
    whitened_class_covariance = whitening @ class_covariance @ whitening.T

    class_eigenvalues, class_eigenvectors = np.linalg.eigh(whitened_class_covariance)
    class_order = np.argsort(class_eigenvalues)[::-1]
    csp_filters = class_eigenvectors[:, class_order].T @ whitening

    half = n_components // 2
    selected_indices = list(range(half)) + list(range(-half, 0))
    return csp_filters[selected_indices]


def fit_ovr_csp(
    x_train: np.ndarray,
    labels_train: np.ndarray,
    n_classes: int,
    n_components: int,
) -> list[np.ndarray]:
    squeezed = _squeeze_trials(x_train)
    return [
        _fit_binary_csp_filters(squeezed, labels_train == class_index, n_components)
        for class_index in range(n_classes)
    ]


def transform_ovr_csp(x: np.ndarray, filters_by_class: list[np.ndarray]) -> np.ndarray:
    squeezed = _squeeze_trials(x)
    feature_blocks: list[np.ndarray] = []
    for filters in filters_by_class:
        block = np.zeros((squeezed.shape[0], filters.shape[0]), dtype=np.float32)
        for trial_index, trial in enumerate(squeezed):
            projected = filters @ trial
            variances = np.var(projected, axis=1)
            block[trial_index] = np.log(
                variances / (np.sum(variances) + 1e-12) + 1e-12
            )
        feature_blocks.append(block)
    return np.concatenate(feature_blocks, axis=1)


def _write_baseline_summary(
    config: RunConfig,
    dataset: DatasetBundle,
    metrics: dict,
    feature_shapes: dict[str, list[int]],
    destination: str | Path,
) -> None:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "# CSP + LDA Baseline Summary",
                "",
                f"- Dataset file: `{config.data_path}`",
                f"- Validation strategy: `{config.validation_strategy}`",
                f"- CSP strategy: one-vs-rest with {config.baseline_csp_components} filters per class",
                f"- LDA solver: `{config.baseline_lda_solver}`",
                f"- LDA shrinkage: `{config.baseline_lda_shrinkage}`",
                f"- Train feature shape: {tuple(feature_shapes['train'])}",
                f"- Valid feature shape: {tuple(feature_shapes['valid'])}",
                f"- Test feature shape: {tuple(feature_shapes['test'])}",
                f"- Train accuracy: {metrics['train']['accuracy']:.4f}",
                f"- Validation accuracy: {metrics['valid']['accuracy']:.4f}",
                f"- Test accuracy: {metrics['test']['accuracy']:.4f}",
                f"- Test macro F1: {metrics['test']['f1_macro']:.4f}",
                "",
                "Interpretation note:",
                "This baseline uses a standard classical EEG decoding recipe rather than deep learning.",
                "It provides a fairer reference point for judging whether EEGNet adds value on this subject split.",
            ]
        ),
        encoding="utf-8",
    )


def run_baseline_experiment(
    config: RunConfig,
    dataset: DatasetBundle | None = None,
) -> dict:
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    Path(config.metrics_dir).mkdir(parents=True, exist_ok=True)

    if dataset is None:
        dataset = prepare_dataset(
            config.data_path,
            validation_ratio=config.validation_ratio,
            random_seed=config.random_seed,
            validation_strategy=config.validation_strategy,
        )

    filters_by_class = fit_ovr_csp(
        dataset.train.x,
        dataset.train.labels,
        dataset.n_classes,
        config.baseline_csp_components,
    )

    feature_splits = {
        "train": transform_ovr_csp(dataset.train.x, filters_by_class),
        "valid": transform_ovr_csp(dataset.valid.x, filters_by_class),
        "test": transform_ovr_csp(dataset.test.x, filters_by_class),
    }

    classifier = LinearDiscriminantAnalysis(
        solver=config.baseline_lda_solver,
        shrinkage=config.baseline_lda_shrinkage,
    )
    classifier.fit(feature_splits["train"], dataset.train.labels)

    probabilities_by_split = {
        split_name: classifier.predict_proba(features)
        for split_name, features in feature_splits.items()
    }
    metrics = evaluate_probabilities_by_split(probabilities_by_split, dataset)

    results = {
        "model": "CSP + LDA",
        "csp_strategy": "one_vs_rest",
        "csp_components_per_class": config.baseline_csp_components,
        "feature_shapes": {
            split_name: list(features.shape)
            for split_name, features in feature_splits.items()
        },
        "metrics": metrics,
    }
    save_json(results, config.metrics_dir / "csp_lda_metrics.json")
    _write_baseline_summary(
        config,
        dataset,
        metrics,
        results["feature_shapes"],
        config.metrics_dir / "csp_lda_summary.md",
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CSP + LDA baseline on the provided BCI IV 2a subject.")
    parser.add_argument(
        "--config",
        default="configs/default_run.json",
        help="Path to the JSON config file.",
    )
    parser.add_argument("--data-path", default=None, help="Override the dataset path from the config.")
    parser.add_argument("--results-dir", default=None, help="Override the results directory from the config.")
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
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    parser.add_argument(
        "--baseline-csp-components",
        type=int,
        default=None,
        help="Override the number of CSP filters per class. Must be even.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    if args.data_path is not None:
        config.data_path = args.data_path
    if args.results_dir is not None:
        config.results_dir = args.results_dir
    if args.validation_ratio is not None:
        config.validation_ratio = args.validation_ratio
    if args.validation_strategy is not None:
        config.validation_strategy = args.validation_strategy
    if args.seed is not None:
        config.random_seed = args.seed
    if args.baseline_csp_components is not None:
        config.baseline_csp_components = args.baseline_csp_components

    run_baseline_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
