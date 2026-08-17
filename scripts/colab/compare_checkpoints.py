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
import torch  # noqa: E402

from sssumo.data import SyntheticDataset  # noqa: E402
from sssumo.models import TDNNDetector  # noqa: E402
from sssumo.training import (NOISE_CONDITIONS, REFRACTORY_CONDITIONS,  # noqa: E402
                             default_dataset_paths)
from sssumo.utils import (Config, calculate_and_log_metrics_synthetic,  # noqa: E402
                          evaluate_on_organic_data)

# lower is better for these
LOWER_IS_BETTER = {"Onset_Distance", "Amplitude_MAE_Scaled", "Duration_MAE_Scaled",
                   "Reconstruction_MASE", "Reconstruction_SMAPE", "Reconstruction_MAE_Scaled"}


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
    results = {}

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
                use_only_n_datapoints=args.trials, plot=False)
            res['organic'] = {str(n): {d: dict(m) for d, m in dm.items()}
                              for n, dm in organic.items()}

        results[label] = res

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=1, default=str)
        print(f'\nwrote {args.out}', flush=True)

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

    print('\nCOMPARE OK', flush=True)


if __name__ == '__main__':
    main()
