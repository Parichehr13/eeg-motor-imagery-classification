from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_motor_imagery.baseline import fit_ovr_csp, run_baseline_experiment, transform_ovr_csp
from eeg_motor_imagery.config import RunConfig
from eeg_motor_imagery.data import prepare_dataset

from tests.helpers import create_synthetic_bci_iv2a_mat, make_test_dir


class BaselineTests(unittest.TestCase):
    def test_csp_feature_shapes_match_expected_dimensions(self) -> None:
        test_dir = make_test_dir(ROOT, "baseline_shapes")
        data_path = create_synthetic_bci_iv2a_mat(test_dir / "synthetic_subject.mat")
        dataset = prepare_dataset(data_path, validation_ratio=0.25, random_seed=5)

        filters_by_class = fit_ovr_csp(
            dataset.train.x,
            dataset.train.labels,
            dataset.n_classes,
            n_components=4,
        )
        train_features = transform_ovr_csp(dataset.train.x, filters_by_class)

        self.assertEqual(len(filters_by_class), dataset.n_classes)
        self.assertEqual(train_features.shape[0], dataset.train.x.shape[0])
        self.assertEqual(train_features.shape[1], dataset.n_classes * 4)

    def test_run_baseline_experiment_saves_metrics(self) -> None:
        test_dir = make_test_dir(ROOT, "baseline_run")
        data_path = create_synthetic_bci_iv2a_mat(test_dir / "synthetic_subject.mat")
        config = RunConfig(
            data_path=str(data_path),
            results_dir=str(test_dir / "results"),
            model_dir=str(test_dir / "models"),
            validation_ratio=0.25,
            baseline_csp_components=4,
        )

        result = run_baseline_experiment(config)

        self.assertIn("metrics", result)
        self.assertIn("test", result["metrics"])
        self.assertTrue((config.metrics_dir / "csp_lda_metrics.json").exists())
        self.assertTrue((config.metrics_dir / "csp_lda_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
