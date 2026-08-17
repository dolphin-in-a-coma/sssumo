"""Per-trial diagnosis of a metric gap between two checkpoints on one dataset.

Aggregate metrics say *that* two checkpoints differ; this says *how*. It reports
the per-trial distribution, so a gap driven by a few catastrophic trials is
distinguishable from a uniform shift, and writes a figure of the worst cases.

    python scripts/colab/diagnose_dataset.py --root-dir /content/run \\
        --config configs/config-0423-ModGaussian_ampl.yaml \\
        --checkpoint published=config-0423-ModGaussian_ampl_24.pth \\
        --checkpoint rerun=my-run_24.pth \\
        --dataset crank --snr inf --trials 64 --out-prefix /content/crank
"""

import argparse
import json
import os
import random

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sssumo.data import OrganicDataset  # noqa: E402
from sssumo.models import TDNNDetector  # noqa: E402
from sssumo.training import DATASET_FILES  # noqa: E402
from sssumo.utils import (Config, calculate_reconstruction_metrics,  # noqa: E402
                          organic_data_to_format)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root-dir', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', action='append', required=True, metavar='LABEL=FILENAME')
    p.add_argument('--dataset', default='crank')
    p.add_argument('--snr', default='inf', help="'inf' or a number")
    p.add_argument('--trials', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-prefix', default='/content/diagnose')
    return p.parse_args()


def load(path, config):
    model = TDNNDetector(
        batchnorm=config.batchnorm, dilations=config.dilations,
        channels=config.channels, kernel_sizes=config.kernel_sizes,
        num_layers=config.num_layers, dropout_rate=config.dropout_rate,
    ).to(config.device, config.dtype)
    model.load_state_dict(torch.load(path, map_location=config.device))
    model.eval()
    return model


def main():
    args = parse_args()
    candidates = dict(spec.split('=', 1) for spec in args.checkpoint)
    snr = float('inf') if args.snr == 'inf' else float(args.snr)

    config = Config(args.config, root_dir=args.root_dir)
    config.experiment_name = f'diagnose-{args.dataset}'
    reconstructor = config.reconstruction_model
    path = os.path.join(config.datasets_dir, DATASET_FILES[args.dataset])

    per_trial, traces = {}, {}
    for label, filename in candidates.items():
        model = load(os.path.join(args.root_dir, 'weights', filename), config)

        # identical dataset construction and trial choice for every checkpoint
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        dataset = OrganicDataset(path, snr_distribution=snr, quadratic_mean=1,
                                 low_pass_filter=np.inf, purpose='test',
                                 noise_mode=getattr(config, 'noise_mode', 'gaussian'))
        indices = list(range(len(dataset)))
        if args.trials < len(indices):
            indices = random.sample(indices, args.trials)

        rows, kept = [], {}
        with torch.no_grad():
            for i in indices:
                x, x_clean, _ = dataset[i]
                x = organic_data_to_format(x, config)
                x_clean = organic_data_to_format(x_clean, config)
                y_pred = model(x)
                reconstructed, _ = reconstructor(y_pred)
                metrics = calculate_reconstruction_metrics(x_clean, y_pred, reconstructed)
                rows.append({'trial': i, **{k: float(v) for k, v in metrics.items()}})
                kept[i] = (x_clean.reshape(-1).cpu().numpy(),
                           reconstructed.reshape(-1).cpu().numpy())
        per_trial[label] = rows
        traces[label] = kept
        r2 = np.array([r['Reconstruction_R2'] for r in rows])
        print(f'{label:12s} n={len(r2)}  R2 mean={r2.mean():.4f} median={np.median(r2):.4f} '
              f'min={r2.min():.4f}  frac<0={np.mean(r2 < 0):.3f}', flush=True)

    labels = list(candidates)
    if len(labels) == 2:
        a, b = labels
        by_trial_a = {r['trial']: r for r in per_trial[a]}
        by_trial_b = {r['trial']: r for r in per_trial[b]}
        shared = sorted(set(by_trial_a) & set(by_trial_b))
        deltas = [(t, by_trial_b[t]['Reconstruction_R2'] - by_trial_a[t]['Reconstruction_R2'])
                  for t in shared]
        deltas.sort(key=lambda p: p[1])
        values = np.array([d for _, d in deltas])
        print(f'\nper-trial delta R2 ({b} - {a}), n={len(values)}')
        print(f'  mean={values.mean():+.4f}  median={np.median(values):+.4f}')
        print(f'  worst={values.min():+.4f}  best={values.max():+.4f}')
        print(f'  trials worse by >0.1: {int(np.sum(values < -0.1))}  '
              f'by >0.5: {int(np.sum(values < -0.5))}')
        share = values[values < 0].sum()
        worst_ten = values[:10].sum()
        if share:
            print(f'  the 10 worst trials account for {worst_ten/share:.0%} '
                  f'of the total negative delta')

        worst = [t for t, _ in deltas[:4]]
        fig, axes = plt.subplots(len(worst), 1, figsize=(11, 2.4 * len(worst)), squeeze=False)
        for ax, trial in zip(axes[:, 0], worst):
            clean, rec_a = traces[a][trial]
            _, rec_b = traces[b][trial]
            span = slice(0, min(600, len(clean)))
            t = np.arange(len(clean[span])) / 60
            ax.plot(t, clean[span], label='clean', lw=1)
            ax.plot(t, rec_a[span], '--', lw=1,
                    label=f'{a} (R2={by_trial_a[trial]["Reconstruction_R2"]:.2f})')
            ax.plot(t, rec_b[span], ':', lw=1.4,
                    label=f'{b} (R2={by_trial_b[trial]["Reconstruction_R2"]:.2f})')
            ax.set_title(f'{args.dataset} trial {trial}, SNR {args.snr}', fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=.3)
        axes[-1, 0].set_xlabel('Time (s)')
        fig.tight_layout()
        fig.savefig(f'{args.out_prefix}_worst.png', dpi=130)
        print(f'\nwrote {args.out_prefix}_worst.png', flush=True)

    with open(f'{args.out_prefix}_per_trial.json', 'w') as f:
        json.dump(per_trial, f, indent=1)
    print(f'wrote {args.out_prefix}_per_trial.json', flush=True)


if __name__ == '__main__':
    main()
