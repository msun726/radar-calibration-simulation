"""Synthetic liquid-level calibration. No real radar or farm data is used."""
from pathlib import Path
import argparse
import csv
import json
import time
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def simulate(seed=42, sensors=24, samples=180):
    """Reference readings are assumed exact; sensor error is affine + noise/outliers.

    One row represents a reference collection, NOT a 10-minute telemetry sample.
    Liquid level is in metres, not volume. No actual radar physics is simulated.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for sensor in range(sensors):
        gain = rng.normal(1.06, .035)
        offset = rng.normal(.09, .045)
        for t in range(samples):
            # Each collection sees a different fill level, in chronological order.
            reference = rng.uniform(.15, 2.35)
            abnormal = rng.random() < .045
            error = rng.normal(0, .025) + (rng.normal(0, .32) if abnormal else 0)
            raw = gain * reference + offset + error
            rows.append((sensor, t, reference, raw, int(abnormal)))
    return np.array(rows, dtype=float)


def fit(x, y):
    """Least squares: reference = intercept + slope * raw."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or np.ptp(x) < 1e-10:
        raise ValueError('At least two distinct sensor readings are required')
    slope, intercept = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
    return np.array([slope, intercept])


def predict(model, x):
    return model[0] * np.asarray(x) + model[1]


def fit_filtered(x, y):
    """Illustrative training-only MAD residual filter; not a sensor fault diagnosis."""
    initial = fit(x, y)
    residual = np.asarray(y) - predict(initial, x)
    median = np.median(residual)
    mad_sigma = 1.4826 * np.median(np.abs(residual - median))
    keep = np.abs(residual - median) <= max(3 * mad_sigma, .04)
    if keep.sum() < 3 or np.ptp(np.asarray(x)[keep]) < 1e-8:
        return initial, np.ones(len(x), dtype=bool)
    return fit(np.asarray(x)[keep], np.asarray(y)[keep]), keep


def metrics(y, estimated):
    error = np.asarray(estimated) - np.asarray(y)
    return {'mae_m': float(np.mean(np.abs(error))), 'rmse_m': float(np.sqrt(np.mean(error ** 2)))}


class OnlineOLS:
    """Stable sufficient statistics; exact expanding-history OLS, no forgetting.

    Updating robust filters or changing old records requires a different strategy.
    """
    def __init__(self):
        self.n = 0
        self.mx = self.my = self.sxx = self.sxy = 0.0

    def update(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        if not len(x):
            return
        n = len(x)
        mx, my = float(x.mean()), float(y.mean())
        dx, dy = mx - self.mx, my - self.my
        total = self.n + n
        cross = self.n * n / total
        self.sxx += float(np.sum((x - mx) ** 2)) + dx * dx * cross
        self.sxy += float(np.sum((x - mx) * (y - my))) + dx * dy * cross
        self.mx += dx * n / total
        self.my += dy * n / total
        self.n = total

    def model(self):
        if self.n < 2 or self.sxx < 1e-12:
            raise ValueError('Insufficient variation to fit a line')
        slope = self.sxy / self.sxx
        return np.array([slope, self.my - slope * self.mx])


def write_csv(path, rows):
    rows = list(rows)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benchmark(seed=42, samples=100000, batch=1000, repeats=5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(.1, 2.5, samples)
    y = .93 * x - .07 + rng.normal(0, .025, samples)
    ends = list(range(batch, samples, batch)) + [samples]
    def full():
        return [fit(x[:end], y[:end]) for end in ends]
    def online():
        state, models, start = OnlineOLS(), [], 0
        for end in ends:
            state.update(x[start:end], y[start:end])
            models.append(state.model())
            start = end
        return models
    full(); online()  # Warm-up, excluded from timing.
    timing = {'full': [], 'incremental': []}
    results = {}
    for repeat in range(repeats):
        # Alternate order to reduce systematic first-run bias.
        for name, fn in ([('full', full), ('incremental', online)] if repeat % 2 == 0 else [('incremental', online), ('full', full)]):
            start = time.perf_counter()
            results[name] = fn()
            timing[name].append(time.perf_counter() - start)
    difference = float(np.max(np.abs(np.array(results['full']) - np.array(results['incremental']))))
    assert difference < 1e-8, 'Incremental coefficients differ from full OLS'
    a, b = (statistics.median(timing[k]) for k in ['full', 'incremental'])
    return {'samples': samples, 'updates': len(ends), 'repeats': repeats,
            'full_seconds_median': a, 'incremental_seconds_median': b,
            'speedup': a / b, 'max_coefficient_difference': difference,
            'timings_seconds': timing}


def run(out, seed=42):
    out.mkdir(parents=True, exist_ok=True)
    data = simulate(seed)
    write_csv(out / 'synthetic_readings.csv', [dict(zip(['sensor_id', 'collection_index', 'reference_level_m', 'raw_level_m', 'injected_outlier'], row)) for row in data])
    # Sensors 0..15 are historical donors; 16..23 are entirely unseen installs.
    donors = data[data[:, 0] < 16]
    pooled, _ = fit_filtered(donors[:, 3], donors[:, 2])
    accuracy_rows, predictions, day_rows, quality_rows = [], [], [], []
    for sensor in range(16, 24):
        d = data[data[:, 0] == sensor]
        train, test = d[:60], d[120:]
        # Collections 60..119 are deliberately unused; every method shares holdout.
        ols = fit(train[:, 3], train[:, 2])
        filtered, keep = fit_filtered(train[:, 3], train[:, 2])
        for method, estimate in [('raw', test[:, 3]), ('local_ols', predict(ols, test[:, 3])), ('filtered_ols', predict(filtered, test[:, 3]))]:
            accuracy_rows.append({'sensor_id': sensor, 'method': method, **metrics(test[:, 2], estimate)})
            for row, value in zip(test, estimate):
                predictions.append({'sensor_id': sensor, 'collection_index': int(row[1]), 'method': method, 'reference_level_m': row[2], 'estimated_level_m': value})
        quality_rows.append({'sensor_id': sensor, 'training_count': len(train), 'training_raw_span_m': float(np.ptp(train[:, 3])), 'filtered_fraction': float(1 - keep.mean()), 'training_retained_residual_rmse_m': metrics(train[keep, 2], predict(filtered, train[keep, 3]))['rmse_m']})
        for count in [3, 5, 10, 20, 40, 60]:
            early = d[:count]
            local = fit(early[:, 3], early[:, 2])
            for method, model in [('historical_pooled', pooled), ('new_sensor_only', local)]:
                day_rows.append({'sensor_id': sensor, 'reference_count': count, 'method': method, **metrics(test[:, 2], predict(model, test[:, 3]))})
    write_csv(out / 'accuracy_by_sensor.csv', accuracy_rows)
    write_csv(out / 'holdout_predictions.csv', predictions)
    write_csv(out / 'day_one_by_sensor.csv', day_rows)
    write_csv(out / 'training_quality_indicators.csv', quality_rows)
    summary = {m: {key: float(np.mean([r[key] for r in accuracy_rows if r['method'] == m])) for key in ['mae_m', 'rmse_m']} for m in ['raw', 'local_ols', 'filtered_ols']}
    perf = benchmark(seed)
    report = {'seed': seed, 'data_type': 'synthetic liquid level only', 'held_out_sensors': list(range(16,24)), 'train_indices': '0-59', 'test_indices': '120-179', 'accuracy_mean_across_sensors': summary, 'benchmark': perf}
    (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = list(summary)
    axes[0].bar(['Raw', 'OLS', 'Filtered OLS'], [summary[m]['mae_m'] * 1000 for m in labels], color=['#999999', '#4c78a8', '#59a14f'])
    axes[0].set(ylabel='Mean holdout MAE (mm)', title='A. Calibration accuracy\n8 unseen sensors, 60 reference readings each')
    for method, label in [('historical_pooled', 'Historical pooled model'), ('new_sensor_only', 'New sensor only')]:
        counts = [3, 5, 10, 20, 40, 60]
        values = np.array([[r['mae_m'] * 1000 for r in day_rows if r['method'] == method and r['reference_count'] == n] for n in counts])
        axes[1].plot(counts, values.mean(axis=1), marker='o', label=label)
        axes[1].fill_between(counts, np.percentile(values, 25, axis=1), np.percentile(values, 75, axis=1), alpha=.15)
    axes[1].set(xlabel='Available reference readings', ylabel='Holdout MAE (mm)', title='B. Cold-start calibration\nBand = across-sensor interquartile range')
    axes[1].legend(fontsize=8)
    axes[2].bar(['Full refit', 'Incremental OLS'], [perf['full_seconds_median'], perf['incremental_seconds_median']], color=['#999999', '#4c78a8'])
    axes[2].set(ylabel='Seconds, median of 5 runs', title=f'C. Repeated updates on this machine\n100,000 rows / 100 updates; {perf["speedup"]:.1f}x')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Synthetic sensor calibration experiment — not field validation', fontsize=14)
    fig.tight_layout()
    fig.savefig(out / 'comparison.png', dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args.output, args.seed)
