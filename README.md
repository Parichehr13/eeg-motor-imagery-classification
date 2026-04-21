# EEG Motor Imagery Classification with EEGNet

Compact deep-learning project for 4-class EEG motor imagery classification using EEGNet on BCI Competition IV 2a subject data. The repository includes a reproducible training pipeline, saved evaluation artifacts, and a lightweight spatial-filter analysis for channel-level interpretation.

## Overview

- Task: classify imagined `left hand`, `right hand`, `feet`, and `tongue` movements from single EEG trials.
- Input: preprocessed trials with shape `22 x 256 x 1`.
- Model: EEGNet, a compact convolutional architecture designed for EEG decoding.
- Scope: single-subject classification on the included `data/raw/bci_iv2a_sub-008.mat` file.

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
6. Evaluates train, validation, and test performance.
7. Saves figures, metrics, model summary, and channel-importance outputs.

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
|   `-- train_model.py
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

Run the default training configuration:

```bash
python -m eeg_motor_imagery.train --config configs/default_run.json
```

Equivalent script entry point:

```bash
python scripts/train_model.py --config configs/default_run.json
```

Quick smoke test:

```bash
python -m eeg_motor_imagery.train --config configs/default_run.json --max-epochs 20
```

## Outputs

The default run writes:

- `models/eegnet_best.weights.h5`
- `results/figures/class_histograms.png`
- `results/figures/training_curves.png`
- `results/figures/confusion_matrices.png`
- `results/figures/spatial_filter_importance.png`
- `results/metrics/metrics.json`
- `results/metrics/training_history.csv`
- `results/metrics/training_history.json`
- `results/metrics/channel_importance.csv`
- `results/metrics/model_summary.txt`
- `results/metrics/results_summary.md`
- `results/metrics/run_config.json`

## Example Results

Current default run:

- Test accuracy: `74.65%`
- Test macro F1: `0.7459`
- Validation accuracy: `65.52%`

Per-class metrics, confusion matrices, and training history are saved under `results/metrics` and `results/figures`.

## Spatial-Filter Interpretation

The project extracts the learned weights from EEGNet's depthwise spatial convolution, converts them to absolute values, and averages them across filters to produce a ranked channel-importance view.

This is intended as a compact interpretability aid rather than a causal neuroscientific claim. In the current saved run, the highest-weighted channels are concentrated around central and centro-parietal regions, which is directionally consistent with motor imagery decoding.

## Limitations

- single-subject scope
- no cross-subject evaluation
- no baseline comparison against classical methods
- no scalp topography visualization

## Future Work

- extend to multiple subjects from BCI Competition IV 2a
- add a simple CSP + classical classifier baseline
- add topographic visualization of channel importance
- add lightweight unit tests for data loading and config handling
