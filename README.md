# EEG Motor Imagery Classification with EEGNet and CSP + LDA

Compact EEG decoding project for 4-class motor imagery classification on BCI Competition IV 2a subject data. The repository compares a compact deep-learning model (EEGNet) against a standard classical baseline (CSP + LDA), saves reproducible evaluation artifacts, and includes lightweight tests plus cautious channel-level interpretation.

## Overview

- Task: classify imagined `left hand`, `right hand`, `feet`, and `tongue` movements from single EEG trials.
- Input: preprocessed trials with shape `22 x 256 x 1`.
- Models: EEGNet and a CSP + LDA baseline.
- Scope: single-subject classification on the included `data/raw/bci_iv2a_sub-008.mat` file using the predefined train/test sessions plus a stratified validation split from the training session.

## Dataset

The repository includes one preprocessed subject file from **BCI Competition IV 2a**:

- 22 EEG channels
- 256 time samples per trial
- 4 motor imagery classes
- separate training and evaluation sessions already defined in the file

## Pipeline

The training workflow:

1. Loads the training and test sessions from the subject `.mat` file.
2. Reshapes each trial to `channels x time x 1`.
3. Builds a stratified validation split from the training session.
4. Standardizes all inputs using training-set statistics only.
5. Trains EEGNet with checkpointing, early stopping, and learning-rate reduction on validation plateaus.
6. Fits a CSP + LDA baseline on the exact same train/validation/test split.
7. Evaluates train, validation, and test performance for both models.
8. Saves figures, metrics, comparison tables, model summary, and channel-importance outputs.

## Repository Structure

```text
.
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- .gitignore
|-- configs/
|   `-- default_run.json
|-- data/
|   `-- raw/
|       `-- bci_iv2a_sub-008.mat
|-- models/
|   `-- eegnet_best.weights.h5
|-- results/
|   |-- figures/
|   `-- metrics/
|-- scripts/
|   |-- run_baseline.py
|   `-- train_model.py
|-- tests/
`-- src/
    `-- eeg_motor_imagery/
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Usage

Run the default experiment configuration:

```bash
python -m eeg_motor_imagery.train --config configs/default_run.json
```

This runs EEGNet, the CSP + LDA baseline, and writes a shared comparison table under `results/metrics/`.

Equivalent script entry point:

```bash
python scripts/train_model.py --config configs/default_run.json
```

Run EEGNet only:

```bash
python -m eeg_motor_imagery.train --config configs/default_run.json --skip-baseline
```

Run the CSP + LDA baseline only:

```bash
python -m eeg_motor_imagery.baseline --config configs/default_run.json
```

Run the lightweight test suite:

```bash
python -m unittest discover -s tests -v
```

## Outputs

The default run writes:

- `models/eegnet_best.weights.h5`
- `results/figures/class_histograms.png`
- `results/figures/training_curves.png`
- `results/figures/confusion_matrices.png`
- `results/figures/spatial_filter_importance.png`
- `results/metrics/metrics.json`
- `results/metrics/eegnet_metrics.json`
- `results/metrics/csp_lda_metrics.json`
- `results/metrics/model_comparison.csv`
- `results/metrics/model_comparison.md`
- `results/metrics/training_history.csv`
- `results/metrics/training_history.json`
- `results/metrics/channel_importance.csv`
- `results/metrics/model_summary.txt`
- `results/metrics/results_summary.md`
- `results/metrics/csp_lda_summary.md`
- `results/metrics/run_config.json`

## Example Results

Current saved default run on `sub-008`:

| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 | Test macro precision | Test macro recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EEGNet | `65.52%` | `0.6610` | `74.65%` | `0.7459` | `0.7575` | `0.7465` |
| CSP + LDA | `75.86%` | `0.7497` | `81.60%` | `0.8178` | `0.8248` | `0.8160` |

The comparison table is saved in machine-readable form at `results/metrics/model_comparison.csv`.

This baseline matters because motor imagery EEG is often evaluated with CSP-based classical pipelines. Reporting both models makes it easier to judge whether EEGNet is genuinely adding value on this split. In the current saved run, the classical baseline outperforms EEGNet, which is a useful and scientifically honest result.

## Spatial-Filter Interpretation

The project extracts the learned weights from EEGNet's depthwise spatial convolution, converts them to absolute values, and averages them across filters to produce a ranked channel-importance view.

This is intended as a compact interpretability aid rather than a causal neuroscientific claim. In the current saved run, the highest-weighted channels are concentrated around central and centro-parietal regions, which is directionally consistent with motor imagery decoding.

## Limitations

- single-subject scope
- no cross-subject evaluation
- one predefined subject split rather than repeated resampling or nested cross-validation
- interpretability is limited to spatial-filter weight inspection rather than source-localized neurophysiological analysis
- no scalp topography visualization yet

## Future Work

- extend to multiple subjects from BCI Competition IV 2a
- compare against additional classical baselines such as filter-bank CSP or Riemannian pipelines
- add topographic visualization of channel importance with electrode coordinates
- add multi-subject aggregate reporting with mean and standard deviation across subjects
