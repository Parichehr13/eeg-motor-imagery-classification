# Experiment Comparison

The project kept the model architecture fixed and only tested a very small set of practical training changes.

| Run | Validation split | Optimizer | Test accuracy | Test macro F1 |
| --- | --- | --- | ---: | ---: |
| `baseline_first_block` | first 10% of train session | SGD | 0.4792 | 0.4421 |
| `sgd_stratified` | stratified shuffled 10% | SGD + callbacks | 0.4688 | 0.4361 |
| `adam_stratified` | stratified shuffled 10% | Adam + callbacks | 0.7465 | 0.7459 |

Notes:

- `baseline_first_block` is the original exercise-style setup promoted from the earlier standalone cleanup.
- `sgd_stratified` improved split quality and training stability, but it did not improve the held-out test set.
- `adam_stratified` was the best modest improvement and is now the default project configuration.
- This is still a single-subject result on the provided file only, so the numbers should be presented as a project outcome rather than a general EEG benchmark claim.
