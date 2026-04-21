from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_motor_imagery.config import RunConfig
from eeg_motor_imagery.train import run_experiment

from tests.helpers import create_synthetic_bci_iv2a_mat, make_test_dir


class PipelineSmokeTests(unittest.TestCase):
    def test_full_pipeline_smoke_run_writes_key_artifacts(self) -> None:
        test_dir = make_test_dir(ROOT, "pipeline")
        data_path = create_synthetic_bci_iv2a_mat(test_dir / "synthetic_subject.mat")
        config = RunConfig(
            data_path=str(data_path),
            results_dir=str(test_dir / "results"),
            model_dir=str(test_dir / "models"),
            validation_ratio=0.25,
            batch_size=4,
            max_epochs=1,
            early_stopping_patience=0,
            reduce_lr_patience=0,
            run_baseline=True,
        )

        result = run_experiment(config)

        self.assertIn("eegnet", result["metrics"])
        self.assertIn("baseline", result["metrics"])
        self.assertTrue((config.metrics_dir / "metrics.json").exists())
        self.assertTrue((config.metrics_dir / "eegnet_metrics.json").exists())
        self.assertTrue((config.metrics_dir / "csp_lda_metrics.json").exists())
        self.assertTrue((config.metrics_dir / "model_comparison.csv").exists())
        self.assertTrue((config.figures_dir / "training_curves.png").exists())


if __name__ == "__main__":
    unittest.main()
