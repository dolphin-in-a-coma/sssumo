"""Compare two or more checkpoints on the same data, paired.

The global RNGs are reseeded identically before each checkpoint, so every model
sees the same synthetic trials, the same organic trial subsample and the same
noise realisations. Without that, small differences are indistinguishable from
sampling variance.

Synthetic conditions have ground truth, so they give onset precision/recall and
amplitude/duration error. Organic uses the TEST participants.

    python scripts/colab/compare_checkpoints.py \\
        --root-dir /content/run --config configs/config-0423-ModGaussian_ampl.yaml \\
        --checkpoint published=config-0423-ModGaussian_ampl_24.pth \\
        --checkpoint mine=my-run_24.pth \\
        --trials 128 --out /content/compare.json
"""

import argparse
import copy
import json
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
from sssumo.utils import (Config, calculate_and_log_metrics_synthetic,  # noqa: E402
                          evaluate_on_organic_data, hierarchical_bootstrap_metrics)

# lower is better for these
LOWER_IS_BETTER = {"Onset_Distance", "Amplitude_MAE_Scaled", "Duration_MAE_Scaled",
                   "Reconstruction_MASE", "Reconstruction_SMAPE", "Reconstruction_MAE_Scaled"}


def bootstrap_frame(noise2dataset_metrics):
    """Nested {noise: {dataset: {metric: [per-trial...]}}} -> long DataFrame."""
    rows = []
    for noise, per_dataset in noise2dataset_metrics.items():
        for dataset, metrics in per_dataset.items():
            if 'Participant' not in metrics:
                continue
            for i in range(len(metrics['Participant'])):
                row = {'Dataset': dataset, 'Noise_Condition': str(noise)}
                row.update({k: v[i] for k, v in metrics.items()})
                rows.append(row)
    return pd.DataFrame(rows)


KEYS = ['Dataset', 'Noise_Condition', 'Participant', 'Trial']


def paired_delta(reference, other):
    """Per-trial `other - reference`, matched on trial identity."""
    merged = reference.merge(other, on=KEYS, suffixes=('_ref', '_new'))
    delta = merged[KEYS].copy()
    metrics = [c for c in reference.columns if c not in KEYS]
    for m in metrics:
        if f'{m}_ref' in merged and pd.api.types.is_numeric_dtype(merged[f'{m}_ref']):
            delta[m] = merged[f'{m}_new'] - merged[f'{m}_ref']
    return delta


def report_intervals(delta, label, reference_label, n_simulations, metrics=None):
    """Bootstrap the paired delta over participants, per dataset and noise level.

    hierarchical_bootstrap_metrics only implements grouping for 'Noise_Condition';
    any other group_by_column silently pools everything. So datasets are split
    here and noise grouping is delegated, giving one interval per dataset x noise.

    sample_participants=True is essential: the default resamples rows within a
    dataset, ignoring participant clustering, which yields intervals that are far
    too narrow. sample_datasets stays False -- the datasets are the population of
    interest, not a sample from one.
    """
    print(f"\n--- paired delta: {label} - {reference_label} "
          f"({n_simulations} resamples, participants resampled) ---", flush=True)
    frames = []
    for dataset in sorted(d for d in delta['Dataset'].unique() if d != 'synthetic'):
        subset = delta[delta['Dataset'] == dataset]
        n_participants = subset['Participant'].nunique()
        result = hierarchical_bootstrap_metrics(
            subset, n_simulations=n_simulations,
            sample_participants=True, sample_datasets=False,
            balance_datasets=False, balance_participants=False,
            group_by_column='Noise_Condition', datasets_to_exclude=[],
            datasets_to_include=[dataset], central_tendency='mean')
        result = result.copy()
        result['Dataset'] = dataset
        result['N_Participants'] = n_participants
        if metrics is not None:
            result = result[result['Metric'].isin(metrics)]
        frames.append(result)
    combined = pd.concat(frames, ignore_index=True)
    combined['Excludes_0'] = ~((combined['Lower_CI'] <= 0) & (combined['Upper_CI'] >= 0))
    cols = ['Dataset', 'Noise_Condition', 'Metric', 'Mean', 'Lower_CI', 'Upper_CI',
            'N_Participants', 'Excludes_0']
    cols = [c for c in cols if c in combined.columns]
    print(combined[cols].to_string(index=False), flush=True)
    n_sig = int(combined['Excludes_0'].sum())
    print(f"  intervals excluding zero: {n_sig} of {len(combined)}", flush=True)
    return combined


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root-dir', required=True, help='holds data/ and weights/')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', action='append', required=True, metavar='LABEL=FILENAME',
                   help='repeatable; FILENAME is relative to <root-dir>/weights/')
    p.add_argument('--trials', type=int, default=128,
                   help='organic trials per dataset; keep above ~32 (see README)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--bootstrap', type=int, default=0, metavar='N',
                   help='bootstrap the PAIRED per-trial difference against the first '
                        'checkpoint over N resamples, resampling participants. 0 disables. '
                        'Pairing is exact because every checkpoint sees identically seeded '
                        'trials, which removes trial-level variance from the interval.')
    p.add_argument('--bootstrap-metrics', nargs='*',
                   default=['Reconstruction_R2', 'Number_of_submovements_per_second'],
                   help='which metrics to report intervals for; empty means all')
    p.add_argument('--skip-synthetic', action='store_true')
    p.add_argument('--skip-organic', action='store_true')
    p.add_argument('--out', default=None, help='write raw metrics as JSON')
    return p.parse_args()


def load(path, config):
    model = TDNNDetector(
        batchnorm=config.batchnorm, dilations=config.dilations,
        channels=config.channels, kernel_sizes=config.kernel_sizes,
        num_layers=config.num_layers, dropout_rate=config.dropout_rate,
    ).to(config.device, config.dtype)
    state = torch.load(path, map_location=config.device)
    model.load_state_dict(state)
    model.eval()
    floats = [v for v in state.values() if v.is_floating_point()]
    assert all(torch.isfinite(v).all() for v in floats), f'{path}: non-finite weights'
    print(f'  loaded {os.path.basename(path)}: {len(state)} tensors, all finite', flush=True)
    return model


def reseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    candidates = dict(spec.split('=', 1) for spec in args.checkpoint)
    results, per_trial = {}, {}

    for label, filename in candidates.items():
        print(f"\n{'='*70}\n{label}\n{'='*70}", flush=True)
        config = Config(args.config, root_dir=args.root_dir)
        config.experiment_name = f'compare-{label}'
        model = load(os.path.join(args.root_dir, 'weights', filename), config)
        reconstructor = config.reconstruction_model
        res = {'synthetic': {}, 'organic': {}}

        if not args.skip_synthetic:
            base = copy.deepcopy(config)
            base.one_sign_chance = base.hard_refractory_chance = 0
            base.easy_refractory_chance = 0
            base.total_duration_distribution = 1000
            base.batch_size = 512
            base.seed = -5
            base.num_samples = 1
            base.refractory_mode = 'percentages'
            for noise in NOISE_CONDITIONS:
                for refractory in REFRACTORY_CONDITIONS:
                    reseed(args.seed)
                    c = copy.deepcopy(base)
                    c.snr_distribution = noise
                    c.refractory_distribution = refractory
                    name = f'snr{noise}_refr{refractory[0]}-{refractory[1]}'
                    dataset = SyntheticDataset(**c.get_dataset_parameters())
                    res['synthetic'][name] = dict(calculate_and_log_metrics_synthetic(
                        dataset, model, c, reconstructor, name))

        if not args.skip_organic:
            reseed(args.seed)
            organic = evaluate_on_organic_data(
                model=model, dataset2path=default_dataset_paths(config),
                noise_conditions=NOISE_CONDITIONS, config=config,
                reconstructor=reconstructor, purpose='test',
                use_only_n_datapoints=args.trials,
                bootstrap_estimate=bool(args.bootstrap), plot=False)
            if args.bootstrap:
                # per-trial values; point estimates are the means of the same data,
                # so the intervals bracket the numbers being reported
                frame = bootstrap_frame(organic)
                per_trial[label] = frame
                metric_cols = [c for c in frame.columns if c not in KEYS]
                res['organic'] = {
                    noise: {d: {m: float(g[m].mean()) for m in metric_cols}
                            for d, g in frame[frame['Noise_Condition'] == noise].groupby('Dataset')}
                    for noise in frame['Noise_Condition'].unique()}
            else:
                res['organic'] = {str(n): {d: dict(m) for d, m in dm.items()}
                                  for n, dm in organic.items()}

        results[label] = res

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=1, default=str)
        print(f'\nwrote {args.out}', flush=True)
        for label, frame in per_trial.items():
            path = f'{args.out}.per_trial.{label}.csv'
            frame.to_csv(path, index=False)
            print(f'wrote {path}', flush=True)

    labels = list(candidates)
    ref = labels[0]
    print(f"\n\n{'='*90}\nCOMPARISON (paired, seed={args.seed}, "
          f"{args.trials} organic trials/dataset)\n{'='*90}", flush=True)

    def report(title, cells):
        if not cells:
            return
        print(f'\n{title}', flush=True)
        header = f"  {'key':46s}" + ''.join(f'{l:>14s}' for l in labels)
        print(header, flush=True)
        for key, values in cells:
            row = f'  {key:46s}' + ''.join(f'{v:14.4f}' for v in values)
            if len(labels) == 2:
                delta = values[1] - values[0]
                better = ('=' if abs(delta) < 1e-9 else
                          labels[1] if (delta < 0) == (key.split('/')[-1].strip()
                                                       in LOWER_IS_BETTER) else labels[0])
                row += f'{delta:+12.4f}  {better}'
            print(row, flush=True)

    cells = []
    for cond in results[ref]['synthetic']:
        for metric, value in results[ref]['synthetic'][cond].items():
            if isinstance(value, (int, float)):
                vals = [results[l]['synthetic'][cond].get(metric) for l in labels]
                if all(isinstance(v, (int, float)) for v in vals):
                    cells.append((f'{cond} / {metric}', vals))
    report('SYNTHETIC (ground truth known)', cells)

    cells = []
    for noise in results[ref]['organic']:
        for dataset in results[ref]['organic'][noise]:
            for metric, value in results[ref]['organic'][noise][dataset].items():
                if isinstance(value, (int, float)):
                    vals = [results[l]['organic'][noise][dataset].get(metric) for l in labels]
                    if all(isinstance(v, (int, float)) for v in vals):
                        cells.append((f'snr{noise} / {dataset} / {metric}', vals))
    report('ORGANIC TEST PARTICIPANTS', cells)

    if args.bootstrap and len(per_trial) > 1:
        reference = labels[0]
        print(f"\n\n{'='*90}\nPAIRED BOOTSTRAP vs {reference}\n{'='*90}", flush=True)
        for label in labels[1:]:
            delta = paired_delta(per_trial[reference], per_trial[label])
            print(f"\nmatched trials: {len(delta)}", flush=True)
            summary = report_intervals(delta, label, reference, args.bootstrap,
                                       metrics=args.bootstrap_metrics)
            if args.out:
                summary.to_csv(f'{args.out}.bootstrap.{label}.csv', index=False)

    print('\nCOMPARE OK', flush=True)


if __name__ == '__main__':
    main()
