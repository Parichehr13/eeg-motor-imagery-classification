# CSP + LDA Baseline Summary

- Dataset file: `data/raw/bci_iv2a_sub-008.mat`
- Validation strategy: `stratified_shuffle`
- CSP strategy: one-vs-rest with 4 filters per class
- LDA solver: `lsqr`
- LDA shrinkage: `auto`
- Train feature shape: (259, 16)
- Valid feature shape: (29, 16)
- Test feature shape: (288, 16)
- Train accuracy: 0.9035
- Validation accuracy: 0.7586
- Test accuracy: 0.8160
- Test macro F1: 0.8178

Interpretation note:
This baseline uses a standard classical EEG decoding recipe rather than deep learning.
It provides a fairer reference point for judging whether EEGNet adds value on this subject split.