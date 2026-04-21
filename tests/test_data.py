from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_motor_imagery.data import prepare_dataset

from tests.helpers import create_synthetic_bci_iv2a_mat, make_test_dir


class DataPreparationTests(unittest.TestCase):
    def test_prepare_dataset_returns_expected_shapes(self) -> None:
        test_dir = make_test_dir(ROOT, "data")
        data_path = create_synthetic_bci_iv2a_mat(test_dir / "synthetic_subject.mat")

        dataset = prepare_dataset(
            data_path,
            validation_ratio=0.25,
            random_seed=11,
            validation_strategy="stratified_shuffle",
        )

        self.assertEqual(dataset.n_classes, 4)
        self.assertEqual(dataset.input_shape, (6, 32, 1))
        self.assertEqual(dataset.train.x.shape[1:], (6, 32, 1))
        self.assertEqual(dataset.valid.x.shape[1:], (6, 32, 1))
        self.assertEqual(dataset.test.x.shape[1:], (6, 32, 1))
        self.assertEqual(dataset.train.y.shape[1], 4)
        self.assertEqual(dataset.valid.y.shape[1], 4)
        self.assertEqual(dataset.test.y.shape[1], 4)
        self.assertEqual(sum(dataset.split_summary["test"].values()), dataset.test.x.shape[0])
        self.assertAlmostEqual(float(dataset.train.x.mean()), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
