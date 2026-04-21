from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_motor_imagery.config import load_config

from tests.helpers import make_test_dir


class ConfigTests(unittest.TestCase):
    def test_load_config_merges_defaults_and_baseline_options(self) -> None:
        test_dir = make_test_dir(ROOT, "config")
        config_path = test_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "data_path": "data/raw/custom_subject.mat",
                    "max_epochs": 12,
                    "baseline_csp_components": 6,
                }
            ),
            encoding="utf-8",
        )

        config = load_config(config_path)

        self.assertEqual(config.data_path, "data/raw/custom_subject.mat")
        self.assertEqual(config.max_epochs, 12)
        self.assertEqual(config.baseline_csp_components, 6)
        self.assertTrue(config.run_baseline)
        self.assertEqual(config.baseline_lda_solver, "lsqr")


if __name__ == "__main__":
    unittest.main()
