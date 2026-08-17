"""Training loop for the detector.

This is the loop from the article, callable from a script or a notebook:

    from sssumo.utils import Config
    from sssumo.training import train

    config = Config('configs/config-0423-ModGaussian_ampl.yaml', root_dir='/content/sssumo')
    config.experiment_name = 'my-run'
    train(config)

Metrics go to wandb when a run is active (`wandb.init(...)` before calling), and
always to `<root_dir>/logs/<experiment_name>.txt`. Checkpoints are written to
`<root_dir>/weights/<experiment_name>_<epoch>.pth` after every epoch.
"""

import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

import wandb

from .data import SyntheticDataset, CombinedSyntheticDataset
from .models import TDNNDetector
from .utils import (
    Config,
    evaluate_on_organic_data,
    evaluate_on_synthetic_data,
    onset_prediction_metrics_on_masks,
)

DATASET_FILES = {
    'steering': 'steering_tangential_velocity_data.csv',
    'crank': 'crank_tangential_velocity_data.csv',
    'Fitts': 'Fitts_tangential_velocity_data.csv',
    'whacamole': 'whacamole_tangential_velocity_data.csv',
    'object_moving': 'object_moving_tangential_velocity_data.csv',
    'pointing': 'pointing_tangential_velocity_data.csv',
    'tablet_writing': 'tablet_writing_tangential_velocity_data.csv',
}

NOISE_CONDITIONS = [float('inf'), 20, 10]
REFRACTORY_CONDITIONS = [(0., 0.5), (0.5, 1.), (1, 1.5)]


def log(message, file):
    with open(file, 'a') as f:
        f.write(message + '\n')
    print(message, flush=True)


def default_dataset_paths(config):
    """The organic datasets, as {name: path} under config.datasets_dir."""
    return {name: os.path.join(config.datasets_dir, filename)
            for name, filename in DATASET_FILES.items()}


def resolve_start_epoch(config, model):
    """Load starting weights per config.start_with_weights, return the epoch reached."""
    latest_epoch = -1
    if not config.start_with_weights or config.start_with_weights == 'Xavier':
        return latest_epoch

    weights_dir = os.path.dirname(config.weights_file)
    weights_file = None

    if isinstance(config.start_with_weights, str):
        weights_file = os.path.join(weights_dir, config.start_with_weights)
    else:
        prefix = os.path.basename(config.weights_file).replace('.pth', '')
        candidates = [f for f in os.listdir(weights_dir) if f.startswith(prefix)]
        if candidates:
            if isinstance(config.start_with_weights, bool):
                latest_epoch = max(int(f.split('_')[-1].replace('.pth', '')) for f in candidates)
            elif isinstance(config.start_with_weights, int):
                latest_epoch = config.start_with_weights
            else:
                raise ValueError(f'Invalid start_with_weights value: {config.start_with_weights}')
            weights_file = config.weights_file.replace('.pth', f'_{latest_epoch}.pth')

    if weights_file is None:
        print('No weights file found, starting from scratch', flush=True)
        return latest_epoch

    model.load_state_dict(torch.load(weights_file, map_location=config.device))
    print(f'Loaded weights from {weights_file}, continuing from epoch {latest_epoch}', flush=True)
    return latest_epoch


def build_organic_statistics_dataset(config, dataset2path, dataset2stats_path, basic_dataset,
                                     model, reconstructor, noise_conditions_train, step,
                                     eval_datapoints=None, plot=False):
    """The semi-supervised mixture: synthetic trials drawn from organic statistics.

    The model is run over the training participants of each organic dataset, the
    detected submovements are pooled into per-dataset statistics, and those become
    conditional distributions a labelled generator samples from. Real signals are
    never targets -- only the shape of their statistics is used.
    """
    dataset2path_train = {k: v for k, v in dataset2path.items() if k in config.datasets}

    evaluate_on_organic_data(
        model=model,
        dataset2path=dataset2path_train,
        noise_conditions=noise_conditions_train,
        config=config,
        reconstructor=reconstructor,
        step=step,
        purpose='train',
        low_pass_filter=np.inf,
        save_pulled_stats=config.datasets_dir,
        use_only_n_datapoints=eval_datapoints,
        plot=plot,
    )

    stats_datasets = [
        SyntheticDataset(joint_distribution=dataset2stats_path[name],
                         **config.get_dataset_parameters())
        for name in config.datasets
    ]

    return CombinedSyntheticDataset(
        stats_datasets + [basic_dataset], proportions=config.proportions,
        total_duration_distribution=config.total_duration_distribution,
        batch_size=config.batch_size, dtype=config.dtype, device=config.device)


def train(config, dataset2path=None, organic_eval_every=5, synthetic_eval_every=1,
          eval_datapoints=None, resume=False, plot=False, seed_offset=0):
    """Train a detector as described by `config`.

    Args:
        config: a `Config`, with `experiment_name` already set.
        dataset2path: {name: csv path} for the organic data; defaults to
            `default_dataset_paths(config)`.
        organic_eval_every: epochs between organic test evaluations; 0 disables.
        synthetic_eval_every: epochs between synthetic evaluations; 0 disables.
            Each one sweeps 9 noise x refractory conditions at 512 trials of 1000
            samples, which is slow without a GPU.
        eval_datapoints: cap on trials per evaluation. `None` uses every trial, as
            the published run did. Small values can leave a pooled-statistics
            column entirely NaN, which raises in `calculate_and_log_metrics_organic`.
        resume: continue from the highest epoch checkpoint in weights/, ignoring
            `config.start_with_weights`.
        plot: draw reconstruction and evaluation figures (useful in a notebook).
        seed_offset: shifts the per-epoch dataset seed. The synthetic stream is
            derived from `dataset.seed`, which is set to the epoch number, so
            `config.seed` does not affect training data -- two runs of the same
            config see identical trials. Use a nonzero offset for a replicate.

    Returns:
        Path of the last checkpoint written, or None if no epoch completed.
    """
    if dataset2path is None:
        dataset2path = default_dataset_paths(config)
    if resume:
        # True makes resolve_start_epoch pick the highest epoch present in weights/
        config.start_with_weights = True

    os.makedirs(os.path.dirname(config.log_file), exist_ok=True)
    os.makedirs(os.path.dirname(config.weights_file), exist_ok=True)
    print(f'Experiment: {config.experiment_name} on {config.device}', flush=True)

    model = TDNNDetector(
        batchnorm=config.batchnorm,
        dilations=config.dilations,
        channels=config.channels,
        kernel_sizes=config.kernel_sizes,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
    ).to(config.device, config.dtype)
    model.eval()

    latest_epoch = resolve_start_epoch(config, model)

    criterion_entropy = nn.BCELoss()
    criterion_entropy_wo_reduction = nn.BCELoss(reduction='none')
    criterion_mse = nn.MSELoss()

    basic_dataset = SyntheticDataset(**config.get_dataset_parameters())
    reconstructor = basic_dataset.reconstruction_model

    train_noise_condition = config.stat_snr_distribution
    noise_conditions_train = [train_noise_condition]
    dataset2stats_path = {
        name: os.path.join(
            config.datasets_dir,
            f'{config.experiment_name}-{name}-{train_noise_condition}-train-pulled_stats.csv')
        for name in DATASET_FILES
    }

    if config.combined_dataset:
        dataset = build_organic_statistics_dataset(
            config, dataset2path, dataset2stats_path, basic_dataset, model, reconstructor,
            noise_conditions_train, step=0, eval_datapoints=eval_datapoints, plot=plot)
    else:
        dataset = basic_dataset
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler_start = config.lr_decay_start
    scheduler_end = config.lr_decay_end
    step_decay = config.lr_decay_total_change ** (
        1 / ((scheduler_end - scheduler_start) * len(dataloader)))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=step_decay)

    model.train()
    if config.start_with_weights == 'Xavier':
        for param in model.parameters():
            if isinstance(param, nn.Conv1d) or isinstance(param, nn.Linear):
                nn.init.xavier_normal_(param)
    tm = time.time()
    checkpoint_path = None

    for epoch in range(latest_epoch + 1, config.num_epochs):
        dataset.seed = epoch + seed_offset  # drives the whole synthetic stream
        detection_loss_mean = 0
        duration_loss_mean = 0
        amplitude_loss_mean = 0
        reconstruction_loss_mean = 0
        reconstruction_detection_loss_mean = 0
        mean_onset_precision = 0
        mean_onset_recall = 0
        mean_onset_distance = 0

        if epoch >= config.reconstruction_loss_start:
            config.use_reconstruction_loss = True

        if epoch >= config.bn_dropout_freeze_start:
            if config.batchnorm:
                for batch_norm in model.batchnorm_layers:
                    batch_norm.eval()
            if config.dropout_rate > 0:
                model.dropout.eval()

        for i, data_ in enumerate(dataloader):
            if epoch >= scheduler_start and epoch < scheduler_end:
                scheduler.step()
            x, x_clean, y = data_
            if x.device != config.device or x.dtype != config.dtype:
                x = x.to(config.device, config.dtype)
                y = y.to(config.device, config.dtype)
            x = x.squeeze(0)
            x_clean = x_clean.squeeze(0)
            y = y.squeeze(0)

            optimizer.zero_grad()
            y_pred = model(x)

            mask = y[:, 0]
            amplitude = y[:, 1]
            duration = y[:, 2]

            mask_pred = y_pred[:, 0]
            amplitude_pred = y_pred[:, 1]
            duration_pred = y_pred[:, 2]
            if y_pred.shape[1] == 4:
                reconstruction_mask_pred = y_pred[:, 3]

            # Compute losses
            if hasattr(config, 'weight_with_amplitude') and config.weight_with_amplitude:
                detection_loss_positives = criterion_entropy_wo_reduction(
                    mask_pred[mask == 1], mask[mask == 1])
                abs_amplitudes = torch.abs(amplitude[mask == 1])
                abs_amplitudes = torch.sqrt(abs_amplitudes)
                abs_amplitudes = abs_amplitudes.detach()
                detection_loss_positives *= abs_amplitudes
                detection_loss_positives = detection_loss_positives.mean() / abs_amplitudes.mean()
            else:
                detection_loss_positives = criterion_entropy(mask_pred[mask == 1], mask[mask == 1])

            if hasattr(config, 'weight_up_to_n_neighbors') and config.weight_up_to_n_neighbors > 0:
                weight_up_to_n_neighbors = config.weight_up_to_n_neighbors
                weighting_factor = 0.5
                scaling_exponent_tensor = torch.zeros_like(mask)
                for j in range(1, weight_up_to_n_neighbors + 1):
                    scaling_exponent_tensor[:, j:] = torch.max(
                        scaling_exponent_tensor[:, j:],
                        mask[:, :-j] * (weight_up_to_n_neighbors - j + 1))
                    scaling_exponent_tensor[:, :-j] = torch.max(
                        scaling_exponent_tensor[:, :-j],
                        mask[:, j:] * (weight_up_to_n_neighbors - j + 1))
                weight_tensor = torch.pow(weighting_factor, scaling_exponent_tensor)
                weight_tensor[mask == 1] = 0
                weight_tensor = weight_tensor.detach()
                detection_loss_negatives = criterion_entropy_wo_reduction(
                    mask_pred[mask == 0], mask[mask == 0]) * weight_tensor[mask == 0]
                detection_loss_negatives = (detection_loss_negatives.mean()
                                            / weight_tensor[mask == 0].mean())
            else:
                detection_loss_negatives = criterion_entropy(mask_pred[mask == 0], mask[mask == 0])

            if hasattr(config, 'negative_loss_multiplier'):
                if config.negative_loss_multiplier in ('adaptive', 'balanced'):
                    binarized_mask_pred = reconstructor.binarizer.apply(mask_pred, False, True)
                    binarized_mask_pred = binarized_mask_pred.detach()
                    pred_to_true_ratio = binarized_mask_pred.sum() / mask.sum()
                    if config.negative_loss_multiplier == 'balanced' and pred_to_true_ratio < 1.0:
                        detection_loss_positives /= pred_to_true_ratio ** 0.5
                        # when there are not enough positives, we punish for false negatives more
                        # but not as much as for false positives
                    pred_to_true_ratio = torch.clamp(pred_to_true_ratio, 1.0, 10.0)
                    detection_loss_negatives *= pred_to_true_ratio
                elif isinstance(config.negative_loss_multiplier, (int, float)):
                    detection_loss_negatives *= config.negative_loss_multiplier
            detection_loss = detection_loss_positives + detection_loss_negatives

            original_duration_loss = criterion_mse(duration_pred[mask == 1], duration[mask == 1])
            original_amplitude_loss = criterion_mse(amplitude_pred[mask == 1], amplitude[mask == 1])

            if y_pred.shape[1] == 4:
                reconstruction_detection_loss_positives = criterion_entropy(
                    reconstruction_mask_pred[mask == 1], mask[mask == 1])
                reconstruction_detection_loss_negatives = criterion_entropy(
                    reconstruction_mask_pred[mask == 0], mask[mask == 0])
                original_reconstruction_detection_loss = (
                    reconstruction_detection_loss_positives
                    + reconstruction_detection_loss_negatives)

                reconstruction_detection_loss_mean *= i / (i + 1)
                reconstruction_detection_loss_mean += (
                    original_reconstruction_detection_loss.item() / (i + 1))
                reconstruction_detection_loss = (original_reconstruction_detection_loss
                                                 / reconstruction_detection_loss_mean
                                                 * detection_loss_mean)
                reconstruction_detection_loss *= 0.1  # decrease the weight

                if wandb.run is not None:
                    wandb.log({'Loss/ReconstructionDetection':
                               original_reconstruction_detection_loss.item()},
                              step=epoch * len(dataloader) + i)

            # Update running means
            detection_loss_mean *= i / (i + 1)
            detection_loss_mean += detection_loss.item() / (i + 1)
            duration_loss_mean *= i / (i + 1)
            duration_loss_mean += original_duration_loss.item() / (i + 1)
            amplitude_loss_mean *= i / (i + 1)
            amplitude_loss_mean += original_amplitude_loss.item() / (i + 1)

            # Normalize the losses onto the scale of the detection loss, so each
            # term contributes comparably regardless of its raw units
            duration_loss = original_duration_loss / duration_loss_mean * detection_loss_mean
            amplitude_loss = original_amplitude_loss / amplitude_loss_mean * detection_loss_mean

            # Compute reconstruction loss if applicable
            reconstructed_x, _ = reconstructor(y_pred)
            if not config.use_reconstruction_loss:
                reconstructed_x = reconstructed_x.detach()
            original_reconstruction_loss = criterion_mse(reconstructed_x, x_clean)
            reconstruction_loss_mean *= i / (i + 1)
            reconstruction_loss_mean += original_reconstruction_loss.item() / (i + 1)
            reconstruction_loss = (original_reconstruction_loss / reconstruction_loss_mean
                                   * detection_loss_mean)

            # Total loss
            loss = detection_loss + duration_loss + amplitude_loss
            if config.use_reconstruction_loss:
                loss += reconstruction_loss
            elif y_pred.shape[1] == 4:
                loss += reconstruction_detection_loss

            loss.backward()
            optimizer.step()

            if wandb.run is not None:
                wandb.log({'Loss/Total': loss.item(),
                           'Loss/Detection': detection_loss.item(),
                           'Loss/Duration': original_duration_loss.item(),
                           'Loss/Amplitude': original_amplitude_loss.item(),
                           'Loss/Reconstruction': original_reconstruction_loss.item()},
                          step=epoch * len(dataloader) + i)

            # Onset metrics and printing progress
            if i % config.log_interval == 0:
                _, _, _, precision, recall, _, distance = onset_prediction_metrics_on_masks(
                    mask, mask_pred)
                mean_onset_precision *= (i // config.log_interval) / (i // config.log_interval + 1)
                mean_onset_precision += precision / (i // config.log_interval + 1)
                mean_onset_recall *= (i // config.log_interval) / (i // config.log_interval + 1)
                mean_onset_recall += recall / (i // config.log_interval + 1)
                mean_onset_distance *= (i // config.log_interval) / (i // config.log_interval + 1)
                mean_onset_distance += distance / (i // config.log_interval + 1)

                if wandb.run is not None:
                    wandb.log({'Onset/Precision': precision,
                               'Onset/Recall': recall,
                               'Onset/Distance': distance,
                               'Params/LearningRate': scheduler.get_last_lr()[0]},
                              step=epoch * len(dataloader) + i)

                message = (f'Epoch {epoch}, Iteration {i}, Loss: {loss.item()},'
                           f'\nDetection Loss: {detection_loss.item()},'
                           f'\nDuration Loss: {original_duration_loss.item()},'
                           f' Amplitude Loss: {original_amplitude_loss.item()},'
                           f'\nReconstruction Loss: {original_reconstruction_loss.item()}'
                           f'\nOnset Precision: {precision}, Onset Recall: {recall},'
                           f' Onset Distance: {distance}'
                           f'\nTime: {time.time() - tm}')
                log(message, config.log_file)

            # Plot reconstructions at intervals
            if (plot or wandb.run is not None) and i % config.plot_interval == 0:
                plt.figure(figsize=(10, 5), dpi=200)
                ids_to_plot = random.sample(range(len(x)), config.reconstructions_to_plot)
                for j, id_ in enumerate(ids_to_plot):
                    num_rows = 2
                    num_cols = math.ceil(config.reconstructions_to_plot / 2)
                    plt.subplot(num_rows, num_cols, j + 1)
                    original_signal = x[id_].squeeze().detach().cpu().numpy()
                    clean_signal = x_clean[id_].squeeze().detach().cpu().numpy()
                    reconstructed_signal = reconstructed_x[id_].squeeze().detach().cpu().numpy()
                    ts = np.arange(len(original_signal)) / 60
                    plt.plot(ts, original_signal)
                    plt.plot(ts, reconstructed_signal, linestyle='--')
                    plt.plot(ts, clean_signal, linestyle=':')
                    if j == 0:
                        plt.legend(['Original', 'Reconstructed', 'Clean'])
                    if j % num_cols == 0:
                        plt.ylabel('Amplitude, a.u.')
                    if j >= num_cols * (num_rows - 1):
                        plt.xlabel('Time (s)')
                    plt.grid()
                plt.suptitle(f'Original vs Reconstructed Signal Examples, '
                             f'Epoch {epoch}, Step {i}')
                if wandb.run is not None:
                    wandb.log({'Reconstructions': wandb.Image(plt)},
                              step=epoch * len(dataloader) + i)
                if plot:
                    plt.show()
                plt.close('all')  # unattended runs have nothing else closing figures

        if wandb.run is not None:
            wandb.log({'Loss_Epoch/Detection_Mean': detection_loss_mean,
                       'Loss_Epoch/Duration_Mean': duration_loss_mean,
                       'Loss_Epoch/Amplitude_Mean': amplitude_loss_mean,
                       'Loss_Epoch/Reconstruction_Mean': reconstruction_loss_mean,
                       'Onset_Epoch/Precision_Mean': mean_onset_precision,
                       'Onset_Epoch/Recall_Mean': mean_onset_recall,
                       'Onset_Epoch/Distance_Mean': mean_onset_distance},
                      step=len(dataloader) * (epoch + 1))

        checkpoint_path = config.weights_file.replace('.pth', f'_{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)

        message = (f'Epoch {epoch} finished,'
                   f'\nDetection Loss: {detection_loss_mean},'
                   f'\nDuration Loss: {duration_loss_mean},'
                   f' Amplitude Loss: {amplitude_loss_mean},'
                   f'\nReconstruction Loss: {reconstruction_loss_mean}'
                   f'\nOnset Precision: {mean_onset_precision},'
                   f' Onset Recall: {mean_onset_recall},'
                   f' Onset Distance: {mean_onset_distance}'
                   f'\nSaved: {checkpoint_path}'
                   f'\nTime: {time.time() - tm}')
        log(message, config.log_file)

        model.eval()
        if synthetic_eval_every and (epoch + 1) % synthetic_eval_every == 0:
            evaluate_on_synthetic_data(
                model=model,
                noise_conditions=NOISE_CONDITIONS,
                refractory_conditions=REFRACTORY_CONDITIONS,
                config=config,
                reconsturctor=reconstructor,
                step=len(dataloader) * (epoch + 1),
            )

        if organic_eval_every and (epoch + 1) % organic_eval_every == 0:
            evaluate_on_organic_data(
                model=model,
                dataset2path=dataset2path,
                noise_conditions=NOISE_CONDITIONS,
                config=config,
                reconstructor=reconstructor,
                step=len(dataloader) * (epoch + 1),
                use_only_n_datapoints=eval_datapoints,
                plot=plot,
            )

        # Refresh the organic-statistics generators from the improved model
        if config.combined_dataset:
            dataset = build_organic_statistics_dataset(
                config, dataset2path, dataset2stats_path, basic_dataset, model, reconstructor,
                noise_conditions_train, step=len(dataloader) * (epoch + 1),
                eval_datapoints=eval_datapoints, plot=plot)
            dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)

        model.train()

    print('Training finished', flush=True)
    return checkpoint_path
