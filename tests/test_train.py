import tempfile
import unittest
from pathlib import Path

import joblib

from ml_pipeline.train import train


class TrainTests(unittest.TestCase):
    def test_reproducibility_and_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            first = train(path)
            second = train(path)
            self.assertEqual(first, second)
            self.assertEqual(first["train_rows"] + first["test_rows"], 2000)
            model = joblib.load(path)["model"]
            probability = model.predict_proba([[12, 55, 2]])[0, 1]
            self.assertTrue(0 <= probability <= 1)
