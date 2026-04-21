# Run Summary

- Validation strategy: `stratified_shuffle`
- Optimizer: `sgd`
- Dataset file: `data/raw/bci_iv2a_sub-008.mat`
- Sampling rate: 128 Hz
- Train/valid/test trials: 259/29/288
- Train split counts: {"left_hand": 65, "right_hand": 65, "feet": 64, "tongue": 65}
- Valid split counts: {"left_hand": 7, "right_hand": 7, "feet": 8, "tongue": 7}
- Standardization mean/std (train only): 0.000000 / 0.000010
- Best epoch by validation loss: 299
- Best validation loss: 1.1328
- Best validation accuracy: 0.4828
- Test accuracy: 0.4688
- Test macro F1: 0.4361
- Top spatially weighted channels: CPz (0.1039), FC1 (0.1000), Cz (0.0957), CP4 (0.0941), CP2 (0.0928)

Interpretation note:
The spatial-filter ranking is a lightweight interpretability view of EEGNet's depthwise spatial layer.
It shows which electrodes received larger absolute weights in this single-subject training run.
These scores are useful for inspection, but they should not be treated as a causal neuroscientific finding.