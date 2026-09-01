"""Cross-evaluate submovement primitives: train on one pulse family, test on another.

Each cell of the matrix takes a detector trained with generator family A and runs it
on synthetic trials rendered with family B, decoding with A -- the situation a
committed method is in when reality generates something else.

The design leans on a property of `SyntheticDataset`: the submovement parameters
(onsets, durations, amplitudes) and the noise are drawn from an RNG stream that does
not depend on the primitive, and only the *rendering* goes through
`reconstruction_model`. Fixing the seed therefore gives every generator family the
identical latent ground truth and the identical noise realisation, differing only in
pulse shape. Cross-family differences are attributable to shape alone.

Three quantities per cell, which separate effects that reconstruction error alone
conflates:

    Reconstruction_R2   decoder A's rendering of its own predictions vs the true
                        signal -- what a fit-quality metric would report
    Onset_F1, ...       recovery of the latent submovements against ground truth
    Oracle_R2           decoder A's rendering of the *true* labels vs the true
                        signal -- the ceiling imposed by shape mismatch alone,
                        with detection error removed

Usage:
    python scripts/colab/cross_evaluate_families.py --root-dir /content/sssumo \
        --epoch 24 --out /content/family_cross_eval.csv
"""

import argparse
import itertools
import os

import matplotlib
matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from sssumo.data import SyntheticDataset  # noqa: E402
from sssumo.models import TDNNDetector  # noqa: E402
from sssumo.models import STEBinarizer  # noqa: E402
from sssumo.utils import (  # noqa: E402
    Config,
    calculate_reconstruction_metrics,
    calculate_supervised_metrics,
    match_onsets_with_predictions,
)

ARMS = ('minjerk', 'gaussian', 'beta_asym', 'lgnb')
NOISE_CONDITIONS = (float('inf'), 20, 10)
REFRACTORY_CONDITIONS = ((0., 0.5), (0.5, 1.), (1., 1.5))


def config_path(arm):
    return f'configs/config-0901-family_{arm}.yaml'


def load_config(arm, root_dir):
    config = Config(config_path(arm), root_dir=root_dir)
    config.experiment_name = f'config-0901-family_{arm}'
    return config


def load_model(config, arm, root_dir, epoch):
    weights = os.path.join(root_dir, 'weights', f'config-0901-family_{arm}_{epoch}.pth')
    if not os.path.exists(weights):
        raise FileNotFoundError(weights)
    model = TDNNDetector(
        batchnorm=config.batchnorm, dilations=config.dilations, channels=config.channels,
        kernel_sizes=config.kernel_sizes, num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
    ).to(config.device, config.dtype)
    model.load_state_dict(torch.load(weights, map_location=config.device))
    model.eval()
    return model


def eval_config(arm, root_dir, batch_size, seed):
    """A generator config pinned to one condition-free, reproducible test stream."""
    config = load_config(arm, root_dir)
    config.one_sign_chance = 0
    config.hard_refractory_chance = 0
    config.easy_refractory_chance = 0
    config.total_duration_distribution = 1000
    config.batch_size = batch_size
    config.seed = seed
    config.num_samples = 1
    config.refractory_mode = 'percentages'
    return config


def signed_onset_bias(mask_true, mask_pred, allowed_distance=5):
    """Mean (predicted onset - true onset) over matched pairs, per trial.

    `Onset_Distance` is unsigned, so it cannot say *which way* a mismatched decoder
    slides its onsets. It should slide: the asymmetric families peak at 0.33 of the
    duration where the symmetric ones peak at 0.50, so a decoder anchored to the
    wrong peak-to-onset lag has a systematic, signed offset. Positive = late.
    """
    binarized = STEBinarizer.apply(mask_pred)
    biases = []
    for i in range(mask_true.shape[0]):
        poses_true = torch.nonzero(mask_true[i:i + 1]).cpu().numpy()
        poses_pred = torch.nonzero(binarized[i:i + 1]).cpu().numpy()
        if len(poses_true) == 0 or len(poses_pred) == 0:
            biases.append(float('nan'))
            continue
        matched = match_onsets_with_predictions(poses_true, poses_pred)
        true_pos, pred_pos = matched[:, 1], matched[:, 2]
        keep = (pred_pos != -1) & (np.abs(true_pos - pred_pos) <= allowed_distance)
        biases.append(float(np.mean(pred_pos[keep] - true_pos[keep])) if keep.any()
                      else float('nan'))
    return biases


def summarise(values):
    """Mean and a 95% normal-approximation interval over trials.

    Some metrics come back as torch tensors still on the device (the submovement
    counts are computed from the mask), and numpy cannot convert a CUDA tensor.
    On CPU this passed silently, so it only surfaced on a GPU run.
    """
    if torch.is_tensor(values):
        values = values.detach().cpu()
    elif isinstance(values, (list, tuple)):
        values = [v.detach().cpu().item() if torch.is_tensor(v) else v for v in values]
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan'), float('nan')
    mean = values.mean()
    half = 1.96 * values.std(ddof=1) / np.sqrt(values.size) if values.size > 1 else 0.0
    return mean, half


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root-dir', required=True)
    parser.add_argument('--epoch', type=int, default=24)
    parser.add_argument('--arms', nargs='+', default=list(ARMS))
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=-5)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    configs = {arm: load_config(arm, args.root_dir) for arm in args.arms}
    models = {arm: load_model(configs[arm], arm, args.root_dir, args.epoch)
              for arm in args.arms}

    rows = []
    for eval_arm in args.arms:
        gen_config = eval_config(eval_arm, args.root_dir, args.batch_size, args.seed)

        for noise, refractory in itertools.product(NOISE_CONDITIONS, REFRACTORY_CONDITIONS):
            gen_config.snr_distribution = noise
            gen_config.refractory_distribution = list(refractory)
            dataset = SyntheticDataset(**gen_config.get_dataset_parameters())
            x, x_clean, y = dataset[0]
            x = x.to(gen_config.device, gen_config.dtype)
            x_clean = x_clean.to(gen_config.device, gen_config.dtype)
            y = y.to(gen_config.device, gen_config.dtype)

            for train_arm in args.arms:
                decoder = configs[train_arm].reconstruction_model
                with torch.no_grad():
                    y_pred = models[train_arm](x)
                    reconstructed_x, _ = decoder(y_pred)
                    # ceiling: the true labels rendered by the decoder's own family
                    oracle_x, _ = decoder(y)

                recon = calculate_reconstruction_metrics(
                    x_clean, y_pred, reconstructed_x, score_for_each_element=True)
                supervised = calculate_supervised_metrics(
                    y, y_pred, score_for_each_element=True)
                oracle = calculate_reconstruction_metrics(
                    x_clean, y, oracle_x, score_for_each_element=True)

                row = {'train_family': train_arm, 'eval_family': eval_arm,
                       'matched': train_arm == eval_arm,
                       'snr': noise, 'overlap': f'{refractory[0]}-{refractory[1]}',
                       'n_trials': args.batch_size}
                for name, metrics in (('', recon), ('', supervised)):
                    for key, value in metrics.items():
                        mean, half = summarise(value)
                        row[f'{name}{key}'] = mean
                        row[f'{name}{key}_ci95'] = half
                mean, half = summarise(oracle['Reconstruction_R2'])
                row['Oracle_R2'], row['Oracle_R2_ci95'] = mean, half
                mean, half = summarise(signed_onset_bias(y[:, 0], y_pred[:, 0]))
                row['Onset_Bias'], row['Onset_Bias_ci95'] = mean, half
                rows.append(row)
                print(f"gen={eval_arm:10s} dec={train_arm:10s} snr={noise:>4} "
                      f"ovl={row['overlap']:8s} "
                      f"recon_R2={row['Reconstruction_R2']:.4f} "
                      f"oracle_R2={row['Oracle_R2']:.4f} "
                      f"onsetF1={row['Onset_F1']:.4f} "
                      f"N/s={row['Number_of_submovements_per_second']:.3f} "
                      f"bias={row['Onset_Bias']:+.3f}",
                      flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out, index=False)
    print(f'\nwrote {args.out}  ({len(frame)} rows)')


if __name__ == '__main__':
    main()
