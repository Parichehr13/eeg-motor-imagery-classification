# Run Summary

- Validation strategy: `stratified_shuffle`
- Optimizer: `adam`
- Dataset file: `data/raw/bci_iv2a_sub-008.mat`
- Sampling rate: 128 Hz
- Train/valid/test trials: 259/29/288
- Train split counts: {"left_hand": 65, "right_hand": 65, "feet": 64, "tongue": 65}
- Valid split counts: {"left_hand": 7, "right_hand": 7, "feet": 8, "tongue": 7}
- Standardization mean/std (train only): 0.000000 / 0.000010
- Best epoch by validation loss: 147
- Best validation loss: 0.7601
- Best validation accuracy: 0.6552
- Test accuracy: 0.7465
- Test macro F1: 0.7459
- Top spatially weighted channels: CP3 (0.1230), CP4 (0.1181), CPz (0.1067), P1 (0.0994), FC1 (0.0976)

Interpretation note:
The spatial-filter ranking is a lightweight interpretability view of EEGNet's depthwise spatial layer.
It shows which electrodes received larger absolute weights in this single-subject training run.
These scores are useful for inspection, but they should not be treated as a causal neuroscientific finding.