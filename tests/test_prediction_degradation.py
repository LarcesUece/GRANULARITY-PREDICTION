"""Regression checks for horizon metrics and selection of the latest run."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('plots', ROOT / 'src' / '02. gerar_gráficos.py')
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


class DegradationTests(unittest.TestCase):
    def test_rmse_uses_samples_axis(self):
        truth = np.zeros((2, 3))
        predictions = np.array([[3, 0, 8], [0, 4, 6]])
        np.testing.assert_allclose(plots.rmse_per_horizon(truth, predictions),
                                   np.sqrt([4.5, 8, 50]))

    def test_bytes_do_not_overflow_float32_square(self):
        truth = np.zeros((2, 3), dtype=np.float32)
        predictions = np.full((2, 3), 1e20, dtype=np.float32)
        result = plots.rmse_per_horizon(truth, predictions)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result, predictions[0])

    def test_rejects_broadcasting_empty_and_nonfinite_inputs(self):
        for truth, prediction in [
            (np.zeros((2, 3)), np.zeros((2, 1))),
            (np.zeros((0, 3)), np.zeros((0, 3))),
            (np.zeros(3), np.zeros(3)),
            (np.zeros((1, 1)), [[np.nan]]),
            ([[np.inf]], np.zeros((1, 1))),
        ]:
            with self.subTest(shape=np.shape(truth)), self.assertRaises(ValueError):
                plots.rmse_per_horizon(truth, prediction)

    def test_latest_metrics_match_overwritten_prediction_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metrics.jsonl'
            rows = [dict(dataset='hour', model='GRU', batch_size=128, score=i)
                    for i in (1, 2)]
            rows.append(dict(dataset='hour', model='LSTM', batch_size=128, score=3))
            path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            result = plots.load_metrics(path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result.loc[result.model == 'GRU', 'score'].item(), 2)

    def test_plot_creates_degradation_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            np.savez(directory / 'example.npz', y_true=np.zeros((4, 12)),
                     y_pred=np.tile(np.linspace(0.1, 0.4, 12), (4, 1)))
            df = plots.pd.DataFrame([dict(dataset='synthetic', model='GRU',
                                         batch_size=128, prediction_path='example.npz')])
            plots.plot_degradation(df, directory, directory)
            self.assertGreater((directory / 'degradacao_synthetic_GRU.png').stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
