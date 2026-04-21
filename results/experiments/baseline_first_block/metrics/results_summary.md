# Run Summary

- Dataset file: `data/raw/bci_iv2a_sub-008.mat`
- Sampling rate: 128 Hz
- Train/valid/test trials: 259/29/288
- Standardization mean/std (train only): 0.000000 / 0.000010
- Test accuracy: 0.4792
- Test macro F1: 0.4421
- Top spatially weighted channels: CPz (0.1072), FC1 (0.1008), P1 (0.0954), CP2 (0.0934), Cz (0.0928)

Interpretation note:
The spatial-filter ranking is a lightweight interpretability view of EEGNet's depthwise spatial layer.
It shows which electrodes received larger absolute weights in this single-subject training run.
These scores are useful for inspection, but they should not be treated as a causal neuroscientific finding.