"""Absolute scores with confidence bands for one or more checkpoints.

Complements compare_checkpoints.py, which reports *differences* between
checkpoints. This reports each checkpoint's own score with an interval, on both
domains.

The resampling unit differs by domain, and using one scheme for both would be
wrong:

* synthetic -- trials are independent draws from the generator, so a plain
  i.i.d. percentile bootstrap over trials is correct.
* organic -- trials cluster within participant, so participants are resampled
  (hierarchical). Treating organic trials as independent would understate the
  interval substantially.

    python scripts/colab/score_checkpoints.py --root-dir /content/run \
        --config configs/config-0423-ModGaussian_ampl.yaml \
        --checkpoint released=config-0423-ModGaussian_ampl_24.pth \
        --bootstrap 2000 --out scores.csv
"""

import argparse
import copy
import os
import random

import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from sssumo.data import SyntheticDataset  # noqa: E402
from sssumo.models import TDNNDetector  # noqa: E402
from sssumo.training import (NOISE_CONDITIONS, REFRACTORY_CONDITIONS,  # noqa: E402
                             default_dataset_paths)
from sssumo.utils import (Config, calculate_reconstruction_metrics,  # noqa: E402
                          calculate_supervised_metrics, evaluate_on_organic_data,
                          hierarchical_bootstrap_metrics)

REPORT = ['Reconstruction_R2', 'Reconstruction_MASE', 'Onset_Precision', 'Onset_Recall',
          'Onset_F1', 'Onset_Distance', 'Amplitude_R2', 'Duration_R2',
          'Number_of_submovements_per_second']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root-dir', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', action='append', required=True, metavar='LABEL=FILENAME')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--trials', type=int, default=128, help='organic trials per dataset')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--skip-synthetic', action='store_true')
    p.add_argument('--skip-organic', action='store_true')
    p.add_argument('--out', required=True, help='tidy CSV of scores and intervals')
    return p.parse_args()


def reseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load(path, config):
    model = TDNNDetector(
        batchnorm=config.batchnorm, dilations=config.dilations,
        channels=config.channels, kernel_sizes=config.kernel_sizes,
        num_layers=config.num_layers, dropout_rate=config.dropout_rate,
    ).to(config.device, config.dtype)
    state = torch.load(path, map_location=config.device)
    model.load_state_dict(state)
    model.eval()
    assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()), path
    return model


def iid_bootstrap(values, n_simulations, confidence=0.95):
    """Percentile bootstrap of the mean for independent observations."""
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return np.nan, np.nan, np.nan, 0
    idx = np.random.randint(0, values.size, size=(n_simulations, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100])
    return values.mean(), lo, hi, values.size


def synthetic_per_trial(model, config, reconstructor, seed):
    """Per-trial synthetic metrics for each noise x refractory condition."""
    base = copy.deepcopy(config)
    base.one_sign_chance = base.hard_refractory_chance = base.easy_refractory_chance = 0
    base.total_duration_distribution = 1000
    base.batch_size = 512
    base.seed = -5
    base.num_samples = 1
    base.refractory_mode = 'percentages'

    rows = []
    for noise in NOISE_CONDITIONS:
        for refractory in REFRACTORY_CONDITIONS:
            reseed(seed)
            c = copy.deepcopy(base)
            c.snr_distribution = noise
            c.refractory_distribution = refractory
            dataset = SyntheticDataset(**c.get_dataset_parameters())
            x, x_clean, y = dataset[0]
            x = x.to(c.device, c.dtype)
            x_clean = x_clean.to(c.device, c.dtype)
            y = y.to(c.device, c.dtype)
            with torch.no_grad():
                y_pred = model(x)
                reconstructed, _ = reconstructor(y_pred)
                rec = calculate_reconstruction_metrics(
                    x_clean, y_pred, reconstructed, score_for_each_element=True)
                sup = calculate_supervised_metrics(y, y_pred, score_for_each_element=True)
            merged = {**rec, **sup}
            n = len(np.atleast_1d(merged['Reconstruction_R2']))
            for i in range(n):
                row = {'Noise': str(noise),
                       'Refractory': f'{refractory[0]}-{refractory[1]}'}
                for key, value in merged.items():
                    array = np.atleast_1d(value)
                    if array.size == n:
                        row[key] = float(array[i])
                rows.append(row)
    return pd.DataFrame(rows)


def organic_per_trial(model, config, reconstructor, seed, trials):
    reseed(seed)
    organic = evaluate_on_organic_data(
        model=model, dataset2path=default_dataset_paths(config),
        noise_conditions=NOISE_CONDITIONS, config=config, reconstructor=reconstructor,
        purpose='test', use_only_n_datapoints=trials, bootstrap_estimate=True, plot=False)
    rows = []
    for noise, per_dataset in organic.items():
        for dataset, metrics in per_dataset.items():
            if dataset == 'synthetic' or 'Participant' not in metrics:
                continue
            for i in range(len(metrics['Participant'])):
                row = {'Dataset': dataset, 'Noise_Condition': str(noise)}
                row.update({k: v[i] for k, v in metrics.items()})
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    candidates = dict(spec.split('=', 1) for spec in args.checkpoint)
    records = []

    for label, filename in candidates.items():
        print(f"\n{'='*70}\n{label}: {filename}\n{'='*70}", flush=True)
        config = Config(args.config, root_dir=args.root_dir)
        config.experiment_name = f'score-{label}'
        model = load(os.path.join(args.root_dir, 'weights', filename), config)
        reconstructor = config.reconstruction_model

        if not args.skip_synthetic:
            frame = synthetic_per_trial(model, config, reconstructor, args.seed)
            # pooled over conditions, and per condition
            groups = [('all', frame)]
            groups += [(f'snr{n}_refr{r}', g) for (n, r), g
                       in frame.groupby(['Noise', 'Refractory'])]
            for name, group in groups:
                for metric in REPORT:
                    if metric not in group:
                        continue
                    mean, lo, hi, n = iid_bootstrap(group[metric], args.bootstrap)
                    records.append(dict(checkpoint=label, domain='synthetic', group=name,
                                        metric=metric, mean=mean, lower=lo, upper=hi,
                                        n_units=n, unit='trial'))
            print(f'  synthetic: {len(frame)} trials over {frame.groupby(["Noise","Refractory"]).ngroups} conditions',
                  flush=True)

        if not args.skip_organic:
            frame = organic_per_trial(model, config, reconstructor, args.seed, args.trials)
            print(f'  organic: {len(frame)} trials, '
                  f'{frame["Participant"].nunique()} participants', flush=True)
            for dataset, group in frame.groupby('Dataset'):
                result = hierarchical_bootstrap_metrics(
                    group, n_simulations=args.bootstrap,
                    sample_participants=True, sample_datasets=False,
                    balance_datasets=False, balance_participants=False,
                    group_by_column='Noise_Condition', datasets_to_exclude=[],
                    datasets_to_include=[dataset], central_tendency='mean')
                n_participants = group['Participant'].nunique()
                for _, row in result.iterrows():
                    if row['Metric'] not in REPORT:
                        continue
                    records.append(dict(
                        checkpoint=label, domain='organic',
                        group=f"{dataset}/snr{row['Noise_Condition']}",
                        metric=row['Metric'], mean=row['Mean'],
                        lower=row['Lower_CI'], upper=row['Upper_CI'],
                        n_units=n_participants, unit='participant'))

    scores = pd.DataFrame(records)
    scores.to_csv(args.out, index=False)
    print(f'\nwrote {args.out} ({len(scores)} rows)', flush=True)

    headline = scores[(scores.group == 'all') |
                      (scores.group.str.startswith('_pooled'))]
    if not headline.empty:
        print('\n=== synthetic, pooled over all 9 conditions ===', flush=True)
        for metric in REPORT:
            sub = headline[headline.metric == metric]
            if sub.empty:
                continue
            print(f'  {metric}', flush=True)
            for _, r in sub.iterrows():
                print(f'    {r.checkpoint:12s} {r["mean"]:8.4f}  '
                      f'[{r.lower:.4f}, {r.upper:.4f}]  n={int(r.n_units)}', flush=True)
    print('\nSCORE OK', flush=True)


if __name__ == '__main__':
    main()
