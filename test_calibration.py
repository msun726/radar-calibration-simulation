import unittest
import numpy as np
from calibration import OnlineOLS, fit, fit_filtered, predict, simulate, metrics


class CalibrationTests(unittest.TestCase):
    def test_recovers_known_mapping(self):
        reference = np.linspace(.1, 2.4, 100)
        raw = 1.08 * reference + .12
        self.assertLess(metrics(reference, predict(fit(raw, reference), raw))['mae_m'], 1e-12)

    def test_online_matches_every_prefix_including_uneven_batches(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=103)
        y = 2 * x + 3 + rng.normal(size=103) * .1
        online = OnlineOLS()
        start = 0
        for end in [2, 9, 51, 100, 103]:
            online.update(x[start:end], y[start:end])
            np.testing.assert_allclose(online.model(), fit(x[:end], y[:end]), atol=1e-10)
            start = end

    def test_degenerate_input_rejected(self):
        with self.assertRaises(ValueError):
            fit([1, 1], [2, 3])
        online = OnlineOLS()
        online.update([1, 1], [2, 3])
        with self.assertRaises(ValueError):
            online.model()

    def test_filter_improves_clean_test_with_contaminated_training(self):
        x = np.linspace(.1, 2.4, 100)
        y = .9 * x + .1
        dirty = y.copy()
        dirty[::10] += 1
        ols = fit(x, dirty)
        robust, keep = fit_filtered(x, dirty)
        self.assertLess(metrics(y, predict(robust, x))['mae_m'], metrics(y, predict(ols, x))['mae_m'])
        self.assertLess(keep.sum(), len(x))

    def test_reproducible_data(self):
        np.testing.assert_array_equal(simulate(42), simulate(42))


if __name__ == '__main__':
    unittest.main()
