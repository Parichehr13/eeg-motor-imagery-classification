# EEG Motor Imagery Classification with EEGNet

This repository turns a course exercise into a small, standalone deep-learning project for classifying single-trial EEG motor imagery using EEGNet. It keeps the scope intentionally modest: one provided subject from BCI Competition IV 2a, one well-known architecture, reproducible training outputs, and a lightweight spatial-filter interpretation.

## Project overview

- **Task:** classify 4 imagined movements from EEG trials: left hand, right hand, feet, and tongue.
- **Input shape:** `22 x 256 x 1` per trial after the provided preprocessing.
- **Model:** EEGNet, a compact convolutional architecture designed for EEG decoding.
- **Dataset scope:** the provided `bci_iv2a_sub-008.mat` subject file only.

This project is honest about its scale: it is not a new research contribution or a multi-subject benchmark. The goal is a clean, reproducible, portfolio-ready implementation of a real EEG classification pipeline.

## Dataset

The included `.mat` file contains one subject from **BCI Competition IV 2a** with preprocessed epochs:

- 22 EEG channels
- 256 time samples per trial
- 4 motor imagery classes
- separate train/test sessions already defined in the file

The original course prompt and lecture notes are preserved in [docs/reference](docs/reference).

## What the pipeline does

The training pipeline reproduces the original exercise steps and adds practical engineering improvements:

1. Loads the provided train/test sessions from the `.mat` file.
2. Reshapes trials to `channels x time x 1`.
3. Builds a stratified validation split from the training session.
4. Standardizes all sets using the training-set mean and standard deviation.
5. Trains EEGNet with a small, stable setup using best-checkpoint saving, early stopping, and learning-rate reduction on validation plateaus.
6. Evaluates train, validation, and test splits.
7. Saves metrics, confusion matrices, training curves, and spatial-filter rankings.

## Repository structure

```text
.
|- CV_SUMMARY.md
|- README.md
|- archive/
|  \- original_solution/
|- configs/
|  \- default_run.json
|- data/
|  \- raw/
|     \- bci_iv2a_sub-008.mat
|- docs/
|  \- reference/
|- models/
|- results/
|  |- figures/
|  |- metrics/
|  \- experiments/
|- scripts/
|  \- train_model.py
\- src/
   \- eeg_motor_imagery/
```

## Installation

Create a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If TensorFlow installation varies on your machine, the rest of the stack is still standard Python ML tooling. The project uses pinned versions to make reruns more consistent.

## How to run

Full training run with the default config:

```bash
python -m eeg_motor_imagery.train --config configs/default_run.json
```

Equivalent script entry point:

```bash
python scripts/train_model.py --config configs/default_run.json
```

Shorter smoke test:

```bash
python -m eeg_motor_imagery.train --config configs/default_run.json --max-epochs 20
```

## Saved outputs

After a successful run, the pipeline writes:

- `models/eegnet_best.weights.h5`: best checkpoint by validation loss
- `results/figures/class_histograms.png`
- `results/figures/training_curves.png`
- `results/figures/confusion_matrices.png`
- `results/figures/spatial_filter_importance.png`
- `results/metrics/training_history.csv`
- `results/metrics/training_history.json`
- `results/metrics/metrics.json`
- `results/metrics/channel_importance.csv`
- `results/metrics/model_summary.txt`
- `results/metrics/results_summary.md`
- `results/metrics/run_config.json`

The `results/experiments/` directory keeps the small comparison runs used during cleanup of the original exercise pipeline.

## Evaluation outputs

The saved metrics include:

- accuracy
- precision
- recall
- F1-score
- per-class classification report
- confusion matrices for train, validation, and test sets

## Spatial-filter interpretation

EEGNet's depthwise spatial convolution acts as a bank of learned spatial filters. This project extracts those filters, takes the absolute values, averages them across learned filters, and ranks channels by their overall importance score.

This is a modest interpretability view, not a causal claim about brain mechanisms. It is still useful for checking whether the network is emphasizing plausible sensor regions, such as central or centro-parietal electrodes often associated with motor imagery decoding.

## Reproducibility features

- fixed random seed
- JSON config file
- automatic directory creation
- stable output file names
- saved training history
- saved best checkpoint
- saved metrics and channel rankings
- no hardcoded absolute dataset paths

Deterministic training can still vary slightly across TensorFlow versions, hardware, and backend kernels. That limitation is normal and is documented rather than hidden.

## Key results

The default config now points to the best modest setting found during cleanup of the original exercise pipeline: stratified validation splitting plus Adam training with early stopping and learning-rate reduction. Run-specific metrics are written to `results/metrics/metrics.json` and summarized in `results/metrics/results_summary.md`.

## Limitations

- single-subject scope using one provided file
- no cross-subject generalization study
- no hyperparameter sweep
- no scalp topography visualization
- no statistical comparison against alternative models

The repository keeps a small comparison between the original exercise-like run and the improved standalone setup under `results/experiments/`, but it is still a compact project rather than a benchmarking study.

## Future improvements

- add multi-subject experiments across the full BCI IV 2a dataset
- compare EEGNet against a simple CSP + classical classifier baseline
- export a scalp topomap for the ranked channels
- add lightweight tests for data loading and config handling
