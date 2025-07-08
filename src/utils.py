
import os

import time
from datetime import datetime
import random
from itertools import product
import collections
from collections import defaultdict
import copy
import yaml
from sklearn.metrics import r2_score

from pathlib import Path
import torch
from typing import Any, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from models import STEBinarizer, STEContinuousReconstructor
from data import OrganicDataset, SyntheticDataset

import wandb

from tqdm import tqdm

# %%

def find_nearest_neighbors(indices_true, indices_pred):
    indices_true_neighbors = np.zeros((len(indices_true), 2), dtype=np.int32) - 1
    indices_pred_neighbors = np.zeros((len(indices_pred), 2), dtype=np.int32) - 1

    i = j = 0
    while i < len(indices_true) and j < len(indices_pred):
        
        in_batch_true, pos_true = indices_true[i]
        in_batch_pred, pos_pred = indices_pred[j]
        # in_batch_pred_next, pos_pred_next = indices_pred[j + 1] if j + 1 < len(indices_pred) else (None, None)

        # skip if not in the same element
        if in_batch_true < in_batch_pred:
            i += 1
            continue
        elif in_batch_true > in_batch_pred:
            j += 1
            continue

        if pos_pred <= pos_true:
            indices_true_neighbors[i][0] = j
            indices_pred_neighbors[j][1] = i
        if pos_pred >= pos_true:
            indices_true_neighbors[i][1] = j
            indices_pred_neighbors[j][0] = i

        if pos_pred <= pos_true:
            j += 1
        if pos_pred >= pos_true:
            i += 1

    return indices_true_neighbors, indices_pred_neighbors

def match_onsets_with_predictions(poses_true, poses_pred, allowed_distance=100):

    # print(poses_true)
    # print(poses_pred)

    neighbors_of_true_id, neighbors_of_pred_id = find_nearest_neighbors(poses_true, poses_pred)

    true_id2pred_id = np.zeros(len(poses_true), dtype=np.int32) - 1

    for true_id, (left_pred_id, right_pred_id) in enumerate(neighbors_of_true_id):
        if left_pred_id == -1 and right_pred_id == -1:
            continue

        true_pos = poses_true[true_id][1]

        if left_pred_id == -1:
            # if there is no left neighbor
            left_pred_pos = -np.inf
        elif neighbors_of_pred_id[left_pred_id][1] != true_id:
            # if the left neighbor's right neighbor is not the current true onset
            left_pred_pos = -np.inf
        else:
            left_pred_pos = poses_pred[left_pred_id][1]

        if right_pred_id == -1:
            # if there is no right neighbor
            right_pred_pos = np.inf
        elif neighbors_of_pred_id[right_pred_id][0] != true_id:
            # if the right neighbor's left neighbor is not the current true onset
            right_pred_pos = np.inf
        else:
            right_pred_pos = poses_pred[right_pred_id][1]

        if right_pred_pos == np.inf and left_pred_pos == -np.inf:
            continue
        elif true_pos - left_pred_pos < right_pred_pos - true_pos:
            true_id2pred_id[true_id] = left_pred_id
        else:
            true_id2pred_id[true_id] = right_pred_id
            neighbors_of_pred_id[right_pred_id][1] = -1 # ensure that the right neighbor is not used again
    poses_true_matched = np.zeros([len(poses_true), 3], dtype=np.int32)

    for true_id, pred_id in enumerate(true_id2pred_id):
        true_in_batch, true_pos = poses_true[true_id]
        if pred_id != -1:
            pred_in_batch, pred_pos = poses_pred[pred_id]
            assert true_in_batch == pred_in_batch, "True and predicted onsets should be in the same element of the batch"
        else:
            pred_pos = -1
        
        poses_true_matched[true_id] = [true_in_batch, true_pos, pred_pos]


    debug = False
    if debug:
        batch_element_numbers = np.unique(poses_true_matched[:, 0])
        for batch_element_number in batch_element_numbers:
            # print(f"Batch Element Number: {batch_element_number}")
            relevant_indices = np.where(poses_true_matched[:, 0] == batch_element_number)[0]
            relevant_pred_poses = poses_true_matched[relevant_indices, 2]
            unique_pred_pos = np.unique(relevant_pred_poses)
            unique_pred_pos = unique_pred_pos[unique_pred_pos != -1]
            non_missed_pred_pos = relevant_pred_poses[relevant_pred_poses != -1]

            if len(unique_pred_pos) != len(non_missed_pred_pos):
                print(f"Batch Element Number: {batch_element_number}")
                # print(f"Relevant Indices: {relevant_indices}")
                print(f"Unique Pred Pos: {len(unique_pred_pos)}")
                print(f"Non Missed Pred Pos: {len(non_missed_pred_pos)}")
                # print counts of each unique pred pos 
                for pred_pos in unique_pred_pos:
                    mask = relevant_pred_poses == pred_pos
                    counts = np.sum(mask)
                    if counts > 2:
                        print(f"Pred Pos: {pred_pos}, Count: {counts}")

                        indices = relevant_indices[mask]
                        print(f"Mathches: {poses_true_matched[indices]}")

    return poses_true_matched


def onset_prediction_metrics(poses_true, poses_pred, allowed_distance=100, amplitude_true=None, amplitude_pred=None):

    poses_true_matched = match_onsets_with_predictions(poses_true, poses_pred)

    # print(poses_true_matched)

    true_pos_vec = poses_true_matched[:, 1]
    pred_pos_vec = poses_true_matched[:, 2]


    if amplitude_true is not None and amplitude_pred is not None:
        amplitude_same_sign = amplitude_true * amplitude_pred > 0
        pred_pos_vec[~amplitude_same_sign] = -1
    

    distances_with_misses = np.abs(true_pos_vec - pred_pos_vec)
    pred_pos_vec[distances_with_misses > allowed_distance] = -1
    
    distances = np.abs(true_pos_vec[pred_pos_vec != -1] - pred_pos_vec[pred_pos_vec != -1])

    true_positives = np.sum(pred_pos_vec != -1)
    false_negatives = np.sum(pred_pos_vec == -1)
    false_positives = len(poses_pred) - true_positives

    if true_positives + false_positives == 0:
        precision = np.nan
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = np.nan
    else:
        recall = true_positives / (true_positives + false_negatives)

    f1_score = 2 * precision * recall / (precision + recall) if not (np.isnan(precision) or np.isnan(recall)) else np.nan

    mean_distance = np.mean(distances) if len(distances) > 0 else np.nan

    return true_positives, false_negatives, false_positives, precision, recall, f1_score, mean_distance
    return precision, recall, mean_distance

def onset_prediction_metrics_on_masks(mask_true, mask_pred, allowed_distance=100, amplitude_true=None, amplitude_pred=None):
    mask_pred = STEBinarizer.apply(mask_pred)
    poses_true = torch.nonzero(mask_true).squeeze(1).cpu().numpy()
    poses_pred = torch.nonzero(mask_pred).squeeze(1).cpu().numpy()

    if amplitude_true is not None and amplitude_pred is not None:
        amplitude_true = amplitude_true[mask_true == 1].cpu().numpy()
        amplitude_pred = amplitude_pred[mask_true == 1].cpu().numpy()

    return onset_prediction_metrics(poses_true, poses_pred, allowed_distance, amplitude_true, amplitude_pred)

def smape(A, F):
    with np.errstate(divide='ignore', invalid='ignore'):
        tmp = 2 * np.abs(F-A) / (np.abs(A) + np.abs(F))
    tmp[np.isnan(tmp)] = 0
    return np.sum(tmp) / len(tmp) * 100

def mase(A, F, hard_mean=0):
    mean_absolute_error = np.mean(np.abs(F - A))
    if hard_mean is not None:
        mean = hard_mean
    else:
        mean = np.mean(A)
    mean_absolute_deviation = np.mean(np.abs(A - mean))
    mean_absolute_deviation = max(mean_absolute_deviation, 1e-5)
    return mean_absolute_error / mean_absolute_deviation


# %%

class Config:
    def __init__(self, config_path: str = "config/config.yaml", root_dir: str | None = None):
        # Load config file
        with Path(config_path).open() as f:
            config = yaml.safe_load(f)


        self._experiment_name = ''
        
        # Update all sections
        for section in config:
            for key, value in config[section].items():
                setattr(self, key, value)
                if key == 'experiment_name':
                    self._experiment_name = value
        
        if root_dir is not None:
            self.root_dir = root_dir

        if self._experiment_name == '':
            config_name = config_path.split('/')[-1].replace('.yaml', '')
            config_name = config_name.replace('config-', '')
            config_name = config_name.lstrip('-_')
            if config_name == '':
                print(f"Warning: There is no experiment name in the config file {config_path}")
            self.experiment_name = config_name 
        
        
        # Override/process specific values
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = getattr(torch, self.dtype)  # Convert string to torch.dtype
        
        # Set paths relative to root
        self.datasets_dir = f'{self.root_dir}/{self.datasets_dir}'
        
        # Initialize reconstruction model
        reconstruction_model_kwargs = {k: getattr(self, k) for k in ['gradient_for_detection', 'device', 'dtype'] if hasattr(self, k)}
        if hasattr(self, 'duration_distribution'):
            if isinstance(self.duration_distribution, list):
                if len(self.duration_distribution) == 2:
                    reconstruction_model_kwargs['duration_range'] = self.duration_distribution
                elif isinstance(self.duration_distribution[0], str) and self.duration_distribution[0] == 'TruncatedLogNormal':
                    reconstruction_model_kwargs['duration_range'] = [self.duration_distribution[3], self.duration_distribution[4]]
        self.reconstruction_model = STEContinuousReconstructor(**reconstruction_model_kwargs)
        
    @property
    def experiment_name(self) -> str:
        return self._experiment_name
    
    @experiment_name.setter
    def experiment_name(self, value: str):
        self._experiment_name = value
        if value:
            self.log_dir = f'{self.root_dir}/TensorBoardLogs/{value}'
            self.weights_file = f'{self.root_dir}/weights/{value}.pth'
            self.log_file = f'{self.root_dir}/logs/{value}.txt'
    
    def get_dataset_parameters(self) -> Dict[str, Any]:
        return {
            k: getattr(self, k)
            for k in [
                'total_duration_distribution',
                'snr_distribution',
                'duration_distribution',
                'amplitude_distribution',
                'amplitude_by_duration_distribution',
                'refractory_distribution',
                'reconstruction_model',
                'max_submovements',
                'num_samples',
                'dtype',
                'device',
                'seed',
                'batch_size',
                'refractory_mode',
                'standardize',
                'one_sign_chance',
                'hard_refractory_chance',
                'easy_refractory_chance',
                'noise_mode',
                'absolute_velocity',
                'make_symmetric',
                'values_require_symmetry'
            ]
            if hasattr(self, k)
        }
    
    def __str__(self):
        return str(self.__dict__)
    
    def to_dict(self):
        return self.__dict__


# %%
# Testing on organic data
def organic_data_to_format(x, config):
    """Format organic data according to the device and dtype."""
    x = torch.nan_to_num(x, 0)
    if x.ndim < 3:
        x = x.unsqueeze(0)
    x = x.to(config.device, config.dtype)
    return x

def calculate_reconstruction_metrics(x_clean, y_pred, reconstructed_x, bootstrap_estimate=False, number_of_bootstrap_samples=1000, score_for_each_element=False, peaks_only=True):
    """Calculate reconstruction metrics for organic data."""

    if not bootstrap_estimate:
        mask_pred = y_pred[:, 0]
        amplitude_pred = y_pred[:, 1]
        duration_pred = y_pred[:, 2]

        if score_for_each_element:

            reconstruction_r2 = []
            reconstruction_mase = []
            reconstruction_smape = []
            number_of_submovements_per_sample = []

            for i in range(x_clean.shape[0]):

                x_clean_np = x_clean[i].reshape(-1).cpu().numpy()
                reconstructed_x_np = reconstructed_x[i].reshape(-1).cpu().numpy()

                reconstruction_r2.append(r2_score(x_clean_np, reconstructed_x_np))
                reconstruction_mase.append(mase(x_clean_np, reconstructed_x_np))
                reconstruction_smape.append(smape(x_clean_np, reconstructed_x_np))
                number_of_submovements_per_sample.append((STEBinarizer.apply(mask_pred, False, peaks_only).sum() / mask_pred.numel()).item())

            reconstruction_r2 = np.array(reconstruction_r2)
            reconstruction_mase = np.array(reconstruction_mase)
            reconstruction_smape = np.array(reconstruction_smape)
            number_of_submovements_per_sample = np.array(number_of_submovements_per_sample)
        else:
            x_clean_np = x_clean.reshape(-1).cpu().numpy()
            reconstructed_x_np = reconstructed_x.reshape(-1).cpu().numpy()
            number_of_submovements_per_sample = (STEBinarizer.apply(mask_pred, False, peaks_only).sum() / mask_pred.numel()).item()
            reconstruction_r2 = r2_score(x_clean_np, reconstructed_x_np)
            reconstruction_mase = mase(x_clean_np, reconstructed_x_np)
            reconstruction_smape = smape(x_clean_np, reconstructed_x_np)

        return {
            'Reconstruction_R2': reconstruction_r2,
            'Reconstruction_MASE': reconstruction_mase,
            'Reconstruction_SMAPE': reconstruction_smape,
            'Number_of_submovements_per_second': number_of_submovements_per_sample * 60
        }
    else:
        metrics_dict = defaultdict(list)

        for i in range(number_of_bootstrap_samples):
            bootstrap_idx = np.random.choice(range(len(x_clean)), size=len(x_clean), replace=True)
            bootstrap_metrics = calculate_reconstruction_metrics(x_clean[bootstrap_idx], y_pred[bootstrap_idx], reconstructed_x[bootstrap_idx], bootstrap_estimate=False)
            for key, value in bootstrap_metrics.items():
                metrics_dict[key].append(value)

        # for key, value in metrics_dict.items():
        #     metrics_dict[key] = [np.mean(value), np.std(value), np.percentile(value, 2.5), np.percentile(value, 97.5)]

        return metrics_dict

def evaluate_organic_trials_for_bootstrap(model, dataset2path, config, reconstructor, 
                            noise_conditions=None, 
                            use_synthetic=True, 
                            bootstrap_estimate=True,
                            use_only_n_datapoints=None,
                            use_only_n_frames=None,
                            purpose='test',
                            output_dir=None,
                            output_filename=None,
                            plot=False,
                            datasets_to_exclude=None,
                            ):
    """
    Evaluates model on organic data and saves metrics to CSV.
    
    Args:
        model: The model to evaluate
        dataset2path: Dictionary mapping dataset names to their paths
        config: Configuration object containing experiment settings
        reconstructor: Reconstructor object
        noise_conditions: List of noise conditions to evaluate (default: [np.inf])
        use_synthetic: Boolean flag for using synthetic data
        bootstrap_estimate: Boolean flag for bootstrap estimation
        use_only_n_datapoints: Number of datapoints to use
        use_only_n_frames: Number of frames to use
    """
    if noise_conditions is None:
        noise_conditions = [np.inf]
        
    dfs = []
    
    with torch.no_grad():
        noise2dataset_metrics = evaluate_on_organic_data(
            model=model,
            dataset2path=dataset2path,
            noise_conditions=noise_conditions,
            config=config,
            reconstructor=reconstructor,
            step=None,
            purpose=purpose,
            low_pass_filter=np.inf,
            save_pulled_stats=None,
            use_synthetic=use_synthetic,
            bootstrap_estimate=bootstrap_estimate,
            use_only_n_datapoints=use_only_n_datapoints,
            use_only_n_frames=use_only_n_frames,
            plot=plot
        )

        # Create dataframes for each noise condition and dataset
        for noise_condition in noise_conditions:
            if isinstance(noise_condition, list):
                noise_condition = '-'.join(map(str, noise_condition))
            for dataset_name in noise2dataset_metrics[noise_condition].keys():
                noise_dataset_metrics_df = pd.DataFrame(
                    noise2dataset_metrics[noise_condition][dataset_name]
                )
                noise_dataset_metrics_df['Noise_Condition'] = noise_condition
                noise_dataset_metrics_df['Dataset'] = dataset_name
                dfs.append(noise_dataset_metrics_df)
        
        # Combine all dataframes and save to CSV
        metrics_df = pd.concat(dfs)
        if output_dir is None:
            try:
                output_dir = config.datasets_dir
            except:
                output_dir = './'
        if output_filename is None:
            datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            try:
                output_filename = f'{config.experiment_name}-organic_metrics_df-{datetime_str}-{purpose}.csv'
            except:
                output_filename = f'organic_metrics_df-{datetime_str}-{purpose}.csv'
        output_path = os.path.join(
            output_dir, 
            output_filename
        )
        metrics_df.to_csv(output_path, index=False)
        
        return metrics_df

def calculate_and_log_metrics_organic(organic_dataset, model, config, reconstructor, dataset_name, noise_condition, step=None, save_pulled_stats=True, purpose='test', for_bootstrap=False, use_only_n_datapoints=None, use_only_n_frames=None, plot=True):
    """Extract metrics for one dataset and log them."""
    datapoints = range(len(organic_dataset))
    
    if use_only_n_datapoints is not None and use_only_n_datapoints < len(datapoints):
        datapoints = random.sample(datapoints, use_only_n_datapoints)

    # reconstruction_r2_mean = 0
    # reconstruction_mase_mean = 0 
    # reconstruction_smape_mean = 0
    # number_of_submovements_per_sample_mean = 0
    if for_bootstrap:
        metric_dict = defaultdict(list)
    else:
        metric_dict = defaultdict(float)

    tm = time.time()
    print(f'Dataset: {dataset_name}, Noise: {noise_condition}, Purpose: {purpose}')

    if purpose != 'test':
        wandb_dataset_name = f'{purpose}/{dataset_name}'
    else:
        wandb_dataset_name = dataset_name

    pulled_statistics = []

    num_samples_to_plot = 10
    num_frames_to_plot = 100
    sr = 60
    n_rows = 5
    n_cols = 2
    plot_indices = set(np.random.choice(datapoints, num_samples_to_plot))
    plot_counter = 0

    if num_samples_to_plot > 0 and plot:
        plt.figure(figsize=(10, 10))

    with torch.no_grad():
        for i in datapoints:
            x, x_clean, _ = organic_dataset[i]

            x = organic_data_to_format(x, config)
            x_clean = organic_data_to_format(x_clean, config)

            if use_only_n_frames is not None and x_clean.shape[-1] > use_only_n_frames:
                first_frame = np.random.randint(0, x_clean.shape[-1] - use_only_n_frames)

                x = x[:, :, first_frame:first_frame + use_only_n_frames]

                x_clean = x_clean[:, :, first_frame:first_frame + use_only_n_frames]
            y_pred = model(x)
            reconstructed_x, _ = reconstructor(y_pred)

            if i in plot_indices and plot:
                first_frame_to_plot = np.random.randint(0, max(1, x_clean.shape[-1] - num_frames_to_plot))
                x_clean_to_plot = x_clean.reshape(-1).cpu().numpy()
                x_clean_to_plot = x_clean_to_plot[first_frame_to_plot:first_frame_to_plot + num_frames_to_plot]
                reconstructed_x_to_plot = reconstructed_x.reshape(-1).cpu().numpy()
                reconstructed_x_to_plot = reconstructed_x_to_plot[first_frame_to_plot:first_frame_to_plot + num_frames_to_plot]
                x_to_plot = x.reshape(-1).cpu().numpy()
                x_to_plot = x_to_plot[first_frame_to_plot:first_frame_to_plot + num_frames_to_plot]

                timestamps = np.arange(0, len(x_clean_to_plot)) / sr
                
                if plot:
                    plt.subplot(n_rows, n_cols, plot_counter + 1)
                    plt.plot(timestamps, x_clean_to_plot, label='Clean')
                    plt.plot(timestamps, reconstructed_x_to_plot, label='Reconstructed')
                    plt.plot(timestamps, x_to_plot, label='Original')
                    if plot_counter == 0:
                        plt.legend()
                        plt.ylabel('Amplitude, a.u.')
                    if plot_counter == num_samples_to_plot - 1:
                        plt.xlabel('Time, s')
                    plot_counter += 1
                    plt.grid()

            metrics = calculate_reconstruction_metrics(x_clean, y_pred, reconstructed_x)
            statistics = extract_statistics(y_pred)
            pulled_statistics.append(statistics)

            # reconstruction_r2_mean += metrics[0] / len(datapoints)
            # reconstruction_mase_mean += metrics[1] / len(datapoints)
            # reconstruction_smape_mean += metrics[2] / len(datapoints)
            # number_of_submovements_per_sample_mean += metrics[3] / len(datapoints)

            for key, value in metrics.items():
                if for_bootstrap:
                    metric_dict[key].append(value)
                else:
                    metric_dict[key] += value / len(datapoints)
            
            if hasattr(organic_dataset, 'trial2participant') and for_bootstrap:
                metric_dict['Participant'].append(organic_dataset.trial2participant[organic_dataset.trials[i]])
            elif for_bootstrap:
                metric_dict['Participant'].append(0)


    if wandb.run is not None:
        if step is not None:
            wandb_keywards = {
                'step': step
            }
    if num_samples_to_plot > 0 and plot:
        plt.suptitle(f'Reconstruction Plots - {wandb_dataset_name}/{noise_condition}', y=1.02)
        plt.tight_layout()
        if wandb.run is not None:
            wandb.log({f'{wandb_dataset_name}/{noise_condition}/reconstruction_plots': wandb.Image(plt)}, **wandb_keywards)
        plt.show()

    if for_bootstrap:
        return metric_dict

    pulled_statistics = pd.concat(pulled_statistics)
    # replace nan with random value from the column
    pulled_statistics['curr_refractory'][pulled_statistics['curr_refractory'] > 100] = np.nan

    for column in pulled_statistics.columns:
        non_nan_values = pulled_statistics[column].dropna()
        pulled_statistics[column] = pulled_statistics[column].apply(lambda x: np.random.choice(non_nan_values) if np.isnan(x) else x)

    for key, value in metric_dict.items():
        print(f'{key}: {value:.4f}')
    print(f'Time: {time.time() - tm:.2f}s')

    # remove outliers, highest and lowest 1%
    outliers_mask = (pulled_statistics.quantile(0.01) <= pulled_statistics) & (pulled_statistics.quantile(0.99) >= pulled_statistics)
    outliers_mask.sum(axis=1) == outliers_mask.shape[1]
    pulled_statistics = pulled_statistics[outliers_mask.all(axis=1)]

    if save_pulled_stats:
        if isinstance(save_pulled_stats, str):
            pulled_statistics.to_csv(f'{save_pulled_stats}/{config.experiment_name}-{dataset_name}-{noise_condition}-{purpose}-pulled_stats.csv')

    if wandb.run is not None:
        wandb.log({f'{wandb_dataset_name}/{noise_condition}/{k}': v for k, v in metric_dict.items()}, **wandb_keywards)

    # # plot histogram of each column of pulled_statistics
    # for column in pulled_statistics.columns:
    #     plt.hist(pulled_statistics[column].dropna(), bins=100)
    #     plt.title(column)
    #     if wandb.run is not None:
    #         wandb.log({f'{dataset_name}/{noise_condition}/{column}': wandb.Image(plt)})
    #     plt.show()

    if plot:
        plt.close('all')
        plt.figure(figsize=(12, 12))
        MAX_BINS = 50
        rice_bins = int(2 * len(pulled_statistics) ** (1/3))
        num_bins = min(MAX_BINS, rice_bins)
        pair_plot = sns.pairplot(pulled_statistics.dropna(), kind='hist', 
                             plot_kws={"cmap": "viridis", 'bins': num_bins}, 
                             diag_kws={"bins": num_bins, 'kde': True})
        
        plt.suptitle(f'Correlation Plot - {wandb_dataset_name}/{noise_condition}', y=1.02)
        
        if wandb.run is not None:
            wandb.log({f'{wandb_dataset_name}/{noise_condition}/correlation_plot': wandb.Image(pair_plot.figure)}, **wandb_keywards)
        plt.show()

    return metric_dict


def evaluate_on_organic_data(model, dataset2path, noise_conditions, config, reconstructor, step=None, purpose='test', low_pass_filter=np.inf, save_pulled_stats=None, use_synthetic=True, bootstrap_estimate=False, use_only_n_datapoints=None, use_only_n_frames=None, plot=True):
    """Evaluate the model on datasets with specified noise conditions."""

    # model.eval()

    if use_synthetic:
        dataset2path['synthetic'] = config.get_dataset_parameters()
        dataset2path['synthetic']['num_samples'] = 128
        dataset2path['synthetic']['batch_size'] = 1
        dataset2path['synthetic']['total_duration_distribution'] = 4000

    noise2dataset_metrics = defaultdict(dict)

    for noise_condition in noise_conditions:
        config.snr_distribution = noise_condition
        organic_dataset_kwargs = {k: getattr(config, k) for k in ['snr_distribution', 'noise_mode', 'absolute_velocity'] if hasattr(config, k)}
        organic_dataset_kwargs.update({'quadratic_mean': 1, 'low_pass_filter': low_pass_filter, 'purpose': purpose})

        dataset_metrics = defaultdict(dict)
        for dataset_name, path in dataset2path.items():
            if dataset_name == 'synthetic':
                synthetic_dataset_kwargs = copy.deepcopy(path)
                synthetic_dataset_kwargs['snr_distribution'] = noise_condition
                organic_dataset = SyntheticDataset(**synthetic_dataset_kwargs)
            else:
                organic_dataset = OrganicDataset(path, **organic_dataset_kwargs)
            metrics_dict = calculate_and_log_metrics_organic(organic_dataset, model, config, reconstructor, dataset_name, noise_condition, step, purpose=purpose, save_pulled_stats=save_pulled_stats, for_bootstrap=bootstrap_estimate, use_only_n_datapoints=use_only_n_datapoints, use_only_n_frames=use_only_n_frames, plot=plot)
        
            dataset_metrics[dataset_name] = metrics_dict

        if isinstance(noise_condition, list):
            noise_condition = '-'.join(map(str, noise_condition))
        noise2dataset_metrics[noise_condition] = dataset_metrics

    return noise2dataset_metrics

# %%
# Testing on synthetic data
def calculate_supervised_metrics(y, y_pred, allowed_distance=5, bootstrap_estimate=False, number_of_bootstrap_samples=1000, score_for_each_element=False):
    """Calculate metrics for supervised learning targets."""


    # Calculate metrics for relevant points
    if not bootstrap_estimate:
        mask = y[:, 0]
        amplitude = y[:, 1]
        duration = y[:, 2]

        mask_pred = y_pred[:, 0]
        amplitude_pred = y_pred[:, 1]
        duration_pred = y_pred[:, 2]

        if score_for_each_element:
            amplitude_r2 = []
            amplitude_mase = []
            amplitude_smape = []
            duration_r2 = []
            duration_mase = []
            duration_smape = []
            onset_precision = []
            onset_recall = []
            onset_f1 = []
            onset_distance = []

            submovement_number_true = []
            submovement_number_correct = []
            submovement_number_predicted = []
            submovement_number_incorrect = []

            for i in range(len(mask)):
                mask_pred_el = mask_pred[i]
                amplitude_pred_el = amplitude_pred[i]
                duration_pred_el = duration_pred[i]
                
                mask_el = mask[i]
                amplitude_el = amplitude[i]
                duration_el = duration[i]

                amplitude_relevant = amplitude_el[mask_el == 1].cpu().numpy()
                amplitude_pred_relevant = amplitude_pred_el[mask_el == 1].cpu().numpy()
                duration_relevant = duration_el[mask_el == 1].cpu().numpy()
                duration_pred_relevant = duration_pred_el[mask_el == 1].cpu().numpy()

                amplitude_r2.append(r2_score(amplitude_relevant, amplitude_pred_relevant))
                amplitude_mase.append(mase(amplitude_relevant, amplitude_pred_relevant))
                amplitude_smape.append(smape(amplitude_relevant, amplitude_pred_relevant))

                duration_r2.append(r2_score(duration_relevant, duration_pred_relevant))
                duration_mase.append(mase(duration_relevant, duration_pred_relevant))
                duration_smape.append(smape(duration_relevant, duration_pred_relevant))

                onset_metrics = onset_prediction_metrics_on_masks(mask[i:i+1], mask_pred[i:i+1], allowed_distance=allowed_distance)
                onset_precision.append(onset_metrics[3])
                onset_recall.append(onset_metrics[4])
                onset_f1.append(onset_metrics[5])
                onset_distance.append(onset_metrics[6])

                submovement_number_true.append(mask_el.sum() / mask_el.numel() * 60)
                submovement_number_correct.append(onset_recall[-1] * submovement_number_true[-1])
                submovement_number_predicted.append(submovement_number_correct[-1] / onset_precision[-1])
                submovement_number_incorrect.append(submovement_number_predicted[-1] - submovement_number_correct[-1])
        else:
            amplitude_relevant = amplitude[mask == 1]
            amplitude_pred_relevant = amplitude_pred[mask == 1]
            duration_relevant = duration[mask == 1]
            duration_pred_relevant = duration_pred[mask == 1]

            # Convert tensors to numpy arrays for metric calculation

            amplitude_relevant_np = amplitude_relevant.cpu().numpy()
            amplitude_pred_relevant_np = amplitude_pred_relevant.cpu().numpy()
            duration_relevant_np = duration_relevant.cpu().numpy()
            duration_pred_relevant_np = duration_pred_relevant.cpu().numpy()

            amplitude_r2 = r2_score(amplitude_relevant_np, amplitude_pred_relevant_np)
            amplitude_mase = mase(amplitude_relevant_np, amplitude_pred_relevant_np)
            amplitude_smape = smape(amplitude_relevant_np, amplitude_pred_relevant_np)

            duration_r2 = r2_score(duration_relevant_np, duration_pred_relevant_np)
            duration_mase = mase(duration_relevant_np, duration_pred_relevant_np)
            duration_smape = smape(duration_relevant_np, duration_pred_relevant_np)

            onset_metrics = onset_prediction_metrics_on_masks(mask, mask_pred, allowed_distance=allowed_distance)
            onset_precision = onset_metrics[3]
            onset_recall = onset_metrics[4]
            onset_f1 = onset_metrics[5]
            onset_distance = onset_metrics[6]

            submovement_number_true = mask.sum() / mask.numel() * 60
            submovement_number_correct = onset_recall * submovement_number_true
            submovement_number_predicted = submovement_number_correct / onset_precision
            submovement_number_incorrect = submovement_number_predicted - submovement_number_correct

        metrics_dict = {
            'Amplitude_R2': amplitude_r2,
            'Amplitude_MAE_Scaled': amplitude_mase,
            'Amplitude_SMAPE': amplitude_smape,
            'Duration_R2': duration_r2,
            'Duration_MAE_Scaled': duration_mase,
            'Duration_SMAPE': duration_smape,
            'Onset_Precision': onset_precision,
            'Onset_Recall': onset_recall,
            'Onset_F1': onset_f1,
            'Onset_Distance': onset_distance,
            'Submovement_Number_True': submovement_number_true,
            'Submovement_Number_Correct': submovement_number_correct,
            'Submovement_Number_Predicted': submovement_number_predicted,
            'Submovement_Number_Incorrect': submovement_number_incorrect
        }
    else:
        metrics_dict = defaultdict(list)

        for i in range(number_of_bootstrap_samples):
            bootstrap_idx = np.random.choice(range(len(y)), size=len(y), replace=True)
            bootstrap_metrics = calculate_supervised_metrics(y[bootstrap_idx], y_pred[bootstrap_idx], bootstrap_estimate=False, allowed_distance=allowed_distance)
            for key, value in bootstrap_metrics.items():
                metrics_dict[key].append(value)

        # for key, value in metrics_dict.items():
        #     metrics_dict[key] = [np.mean(value), np.std(value), np.percentile(value, 2.5), np.percentile(value, 97.5)]

    return metrics_dict


def calculate_and_log_metrics_synthetic(dataset, model, config, reconstructor, condition_name, step=None):
    """Calculate and log metrics for one synthetic dataset."""
    metric_dict = defaultdict(float)
    tm = time.time()
    print(f'Condition: {condition_name}')

    with torch.no_grad():
        for i in range(len(dataset)):
            x, x_clean, y = dataset[i]
            x = x.to(config.device, config.dtype)
            x_clean = x_clean.to(config.device, config.dtype)
            y = y.to(config.device, config.dtype)
            y_pred = model(x)


            reconstructed_x, _ = reconstructor(y_pred)

            reconstruction_metrics = calculate_reconstruction_metrics(x_clean, y_pred, reconstructed_x)
            supervised_metrics = calculate_supervised_metrics(y, y_pred)


            # Accumulate means
            for metrics in [reconstruction_metrics, supervised_metrics]:
                for key, value in metrics.items():
                    metric_dict[key] += value / len(dataset)

    # Calculate F1 score
    # onset_f1_mean = 2 * metric_dict['Onset_Precision'] * metric_dict['Onset_Recall'] / (
    #     metric_dict['Onset_Precision'] + metric_dict['Onset_Recall'])
    # metric_dict['Onset_F1'] = onset_f1_mean

    # metric_dict['Number_of_submovements_per_second'] *= 60
    metric_dict['Amplitude_MAE_Scaled'] *= 100
    metric_dict['Duration_MAE_Scaled'] *= 100
    metric_dict['Reconstruction_MAE_Scaled'] *= 100

    # Print metrics
    for key, value in metric_dict.items():
        print(f'{key}: {value:.4f}')
    print(f'Time: {time.time() - tm:.2f}s')

    # Log to wandb if enabled
    if wandb.run is not None:
        wandb_kwargs = {'step': step} if step is not None else {}
        wandb.log({f'{condition_name}/{k}': v for k, v in metric_dict.items()}, **wandb_kwargs)

    return metric_dict

def evaluate_on_synthetic_data(model, noise_conditions, refractory_conditions, config, reconsturctor=None, step=None, seed=None, force_percentage_refractory=True):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


    config_copy = copy.deepcopy(config)

    config_copy.one_sign_chance = 0
    config_copy.hard_refractory_chance = 0
    config_copy.easy_refractory_chance = 0
    config_copy.total_duration_distribution = 1000
    config_copy.batch_size = 512
    config_copy.seed = -5
    config_copy.num_samples = 1

    if force_percentage_refractory:
        config_copy.refractory_mode = 'percentages'

    if reconsturctor is None:
        reconsturctor = SyntheticDataset(**config_copy.get_dataset_parameters()).reconstruction_model
        # same reconstruction model for all noise and refractory conditions


    for noise_condition, refractory_condition in product(noise_conditions, refractory_conditions):
        config_copy.snr_distribution = noise_condition
        config_copy.refractory_distribution = refractory_condition
        condition_name = f'Overlapping-{refractory_condition[0]}-{refractory_condition[1]}_Noise-{noise_condition}'

        dataset = SyntheticDataset(**config_copy.get_dataset_parameters())

        calculate_and_log_metrics_synthetic(dataset, model, config_copy, reconsturctor, condition_name, step)

def extract_statistics(y_pred):

    assert (y_pred.dim() == 2) or (y_pred.dim() == 3 and y_pred.shape[0] == 1), "y_pred must be 2D or 3D with batch size 1"
    if y_pred.dim() == 2:
        y_pred = y_pred.unsqueeze(0)
    binarized_mask = STEBinarizer.apply(y_pred[:, 0])
    y_pred = y_pred.squeeze()
    binarized_mask = binarized_mask.squeeze()

    onset_index, = torch.where(binarized_mask)

    amplitude = y_pred[1]
    duration = y_pred[2]
    mean_velocity = amplitude / duration

    relevant_amplitude = amplitude[binarized_mask == 1]
    relevant_duration = duration[binarized_mask == 1]
    relevant_mean_velocity = mean_velocity[binarized_mask == 1]

    total_duration = torch.ones_like(onset_index[-1:]) * y_pred.shape[-1]

    refractory_time = onset_index.diff(append=total_duration)
    # refractory_fraction = refractory_time / relevant_duration

    next_duration = torch.roll(relevant_duration, -1)
    next_amplitude = torch.roll(relevant_amplitude, -1)
    next_mean_velocity = torch.roll(relevant_mean_velocity, -1)

    next_duration[-1:] = torch.nan
    next_amplitude[-1:] = torch.nan
    next_mean_velocity[-1:] = torch.nan

    statistics_df = pd.DataFrame({
        'curr_duration': relevant_duration.cpu().numpy(),
        'curr_amplitude': relevant_amplitude.cpu().numpy(),
        'curr_mean_velocity': relevant_mean_velocity.cpu().numpy(),
        'curr_refractory': refractory_time.cpu().numpy(),
        'next_duration': next_duration.cpu().numpy(),
        'next_amplitude': next_amplitude.cpu().numpy(),
        'next_mean_velocity': next_mean_velocity.cpu().numpy()
    })

    return statistics_df

# %%

def hierarchical_bootstrap_metrics(metrics_df, 
                                  n_simulations=1000, 
                                  confidence_level=0.95,
                                  balance_participants=False,
                                  balance_datasets=True,
                                  datasets_to_include=None,
                                  datasets_to_exclude=None,
                                  sample_datasets=False,
                                  sample_participants=False,
                                  group_by_column='Noise_Condition',
                                  save_to=None,
                                  central_tendency='median'
                                  ):
    """
    Performs hierarchical bootstrapping to estimate mean and confidence intervals
    for each metric across datasets and participants.
    
    Args:
        metrics_df: DataFrame containing metrics data with columns for Dataset and Participant
        n_simulations: Number of bootstrap simulations to run
        confidence_level: Confidence level for intervals (default: 0.95)
        balance_participants: If True, sample equal number of participants from each dataset
        balance_datasets: If True, sample datasets with equal probability
        datasets_to_include: List of specific datasets to include (None means all available)
        datasets_to_exclude: List of datasets to exclude (default excludes 'synthetic')
        
    Returns:
        DataFrame with bootstrap results including mean and confidence intervals for each metric
    """

    if isinstance(metrics_df, str):
        metrics_df = pd.read_csv(metrics_df)

    if group_by_column == 'Noise_Condition':
        unique_groups = metrics_df[group_by_column].unique()
        bootstrap_results_list = []
        for group in unique_groups:
            group_df = metrics_df[metrics_df[group_by_column] == group]
            bootstrap_results = hierarchical_bootstrap_metrics(
                group_df,
                n_simulations=n_simulations,
                confidence_level=confidence_level,
                balance_participants=balance_participants,
                balance_datasets=balance_datasets,
                datasets_to_include=datasets_to_include,
                datasets_to_exclude=datasets_to_exclude,
                sample_datasets=sample_datasets,
                sample_participants=sample_participants,
                group_by_column=False
            )
            bootstrap_results['Noise_Condition'] = group
            bootstrap_results_list.append(bootstrap_results)
        bootstrap_results = pd.concat(bootstrap_results_list)

        if save_to is not None:
            pd.DataFrame(bootstrap_results).to_csv(save_to, index=False)
    

        return bootstrap_results
    
    # Set default for datasets_to_exclude if None
    if datasets_to_exclude is None:
        datasets_to_exclude = ['synthetic']
    
    # Filter datasets based on inclusion/exclusion criteria
    available_datasets = metrics_df['Dataset'].unique()
    
    if datasets_to_include is not None:
        selected_datasets = [d for d in datasets_to_include if d in available_datasets]
    else:
        selected_datasets = [d for d in available_datasets if d not in datasets_to_exclude]
    
    # Filter the metrics dataframe to only include selected datasets
    filtered_df = metrics_df[metrics_df['Dataset'].isin(selected_datasets)].copy()
    
    # Identify numeric metric columns (excluding Participant, Dataset, etc.)
    non_metric_cols = ['Participant', 'Dataset', 'Noise_Condition']
    metric_columns = [col for col in filtered_df.columns if col not in non_metric_cols]
    
    # Initialize storage for bootstrap results
    # bootstrap_results = {metric: [] for metric in metric_columns}
    
    # Run bootstrap simulations

    max_datapoints_per_dataset = filtered_df['Dataset'].value_counts().max()


    bootstrap_results = defaultdict(list)

    simulation_values = {metric: [] for metric in metric_columns}
        

    for _ in range(n_simulations): # tqdm(range(n_simulations), desc="Bootstrapping"):

        sampled_dataset_data_list = []
        if sample_datasets:
            sampled_datasets = np.random.choice(selected_datasets, 
                                                size=len(selected_datasets), 
                                                replace=True)
        else:
            sampled_datasets = selected_datasets
        
        # Storage for this simulation's metric values
        # Step 2: For each sampled dataset, sample participants and their metrics
        for dataset in sampled_datasets:
            dataset_data = filtered_df[filtered_df['Dataset'] == dataset].reset_index(drop=True)
            datapoints_per_dataset = dataset_data.shape[0]

            if balance_datasets:
                datapoints_per_dataset = max_datapoints_per_dataset
        
            participants = dataset_data['Participant'].unique()

            if sample_participants:
                sampled_participants = np.random.choice(participants, 
                                                        size=len(participants), 
                                                        replace=True)
            else:
                sampled_participants = participants
            
            participant2weights = collections.Counter(sampled_participants)
            
            if balance_participants:
                participant2frequency = dataset_data['Participant'].value_counts(normalize=True)
                participant2weights = {participant: weight/participant2frequency[participant] for participant, weight in participant2weights.items()} # make less likely participant data more likely to be sampled

            dataset_data['Probability'] = dataset_data['Participant'].map(participant2weights).values
            dataset_data['Probability'] /= dataset_data['Probability'].sum()

            sampled_idx = np.random.choice(dataset_data.index, 
                                        size=datapoints_per_dataset, 
                                        replace=True, 
                                        p=dataset_data['Probability'])
            sampled_dataset_data = dataset_data.iloc[sampled_idx]
            
            sampled_dataset_data_list.append(sampled_dataset_data)
    
        simulated_data = pd.concat(sampled_dataset_data_list)

        for metric in metric_columns:
            if central_tendency == 'median':
                simulation_values[metric].append(simulated_data[metric].median())
            else:
                simulation_values[metric].append(simulated_data[metric].mean())
    
    # Calculate statistics from bootstrap results
    results = []
    alpha = 1 - confidence_level
    
    for metric in metric_columns:
        values = np.array(simulation_values[metric])
        mean = np.mean(values)
        lower_ci = np.percentile(values, alpha/2 * 100)
        upper_ci = np.percentile(values, (1 - alpha/2) * 100)
        
        results.append({
            'Metric': metric,
            'Mean': mean,
            'Lower_CI': lower_ci,
            'Upper_CI': upper_ci,
            'CI_Width': upper_ci - lower_ci,
            'Confidence_Level': confidence_level
        })

    if save_to is not None:
        pd.DataFrame(results).to_csv(save_to, index=False)
    
    return pd.DataFrame(results)

def plot_bootstrap_results(bootstrap_results, figsize=(12, 8)):
    """
    Plots the bootstrap results with error bars for each metric.
    
    Args:
        bootstrap_results: DataFrame from hierarchical_bootstrap_metrics
        figsize: Size of the figure as a tuple (width, height)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=figsize)
    sns.set_style('whitegrid')
    
    # Plot each metric with its confidence interval
    plt.errorbar(
        x=bootstrap_results['Metric'],
        y=bootstrap_results['Mean'],
        yerr=[
            bootstrap_results['Mean'] - bootstrap_results['Lower_CI'],
            bootstrap_results['Upper_CI'] - bootstrap_results['Mean']
        ],
        fmt='o',
        capsize=5,
        ecolor='black',
        markersize=8,
        color='blue',
        alpha=0.7
    )
    
    plt.title(f'Bootstrap Estimates with {bootstrap_results["Confidence_Level"].iloc[0]*100}% Confidence Intervals')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Example usage:
# bootstrap_results = hierarchical_bootstrap_metrics(
#     metrics_df=your_metrics_df,
#     n_simulations=1000,
#     balance_participants=True,
#     balance_datasets=True,
#     datasets_to_include=['steering', 'Fitts', 'pointing']
# )
# 
# print(bootstrap_results)
# plot_bootstrap_results(bootstrap_results)

def wild_cluster_bootstrap_metrics(
    metrics_df,
    n_simulations=1000,
    confidence_level=0.95,
    balance_participants=False,
    balance_datasets=True,
    datasets_to_include=None,
    datasets_to_exclude=None,
    group_by_column='Noise_Condition',
    save_to=None
):
    """
    Wild-cluster bootstrap implementation matching the signature of the twice-nested bootstrap.

    Args:
        metrics_df: DataFrame with columns ['Dataset', 'Participant', ...metric columns...] and optional grouping column.
        n_simulations: Number of bootstrap reps.
        confidence_level: e.g. 0.95 for 95% CI.
        balance_participants: Ignored in wild-cluster bootstrap.
        balance_datasets: Ignored in wild-cluster bootstrap.
        datasets_to_include: List of dataset names to include, or None for all except excluded.
        datasets_to_exclude: List of dataset names to exclude (default ['synthetic']).
        group_by_column: Column name to split by (e.g. 'Noise_Condition').
        save_to: optional filepath to save CSV of results.

    Returns:
        DataFrame with columns ['Metric', 'Mean', 'Lower_CI', 'Upper_CI', 'CI_Width', 'Confidence_Level', (group_by_column)]
    """
    # 1) Handle grouping if requested
    if group_by_column and group_by_column in metrics_df.columns:
        groups = metrics_df[group_by_column].unique()
        all_results = []
        for grp in groups:
            sub_df = metrics_df[metrics_df[group_by_column] == grp]
            res = wild_cluster_bootstrap_metrics(
                sub_df,
                n_simulations=n_simulations,
                confidence_level=confidence_level,
                balance_participants=balance_participants,
                balance_datasets=balance_datasets,
                datasets_to_include=datasets_to_include,
                datasets_to_exclude=datasets_to_exclude,
                group_by_column=False,
                save_to=None
            )
            res[group_by_column] = grp
            all_results.append(res)
        final_df = pd.concat(all_results, ignore_index=True)
        if save_to:
            final_df.to_csv(save_to, index=False)
        return final_df

    # 2) Filter datasets
    if datasets_to_exclude is None:
        datasets_to_exclude = ['synthetic']
    available = metrics_df['Dataset'].unique()
    if datasets_to_include is not None:
        selected = [d for d in datasets_to_include if d in available]
    else:
        selected = [d for d in available if d not in datasets_to_exclude]
    df = metrics_df[metrics_df['Dataset'].isin(selected)].copy()

    # 3) Identify metric columns
    non_metric = ['Dataset', 'Participant', 'Noise_Condition']
    metric_cols = [c for c in df.columns if c not in non_metric and (not isinstance(group_by_column, str) or c != group_by_column)]

    # 4) Prepare results container
    results = []
    alpha = 1 - confidence_level

    # 5) Perform wild-cluster bootstrap for each metric
    for metric in metric_cols:
        # cluster means and residuals
        cl_means = df.groupby('Dataset')[metric].mean()
        overall_mean = cl_means.mean()
        resid = cl_means - overall_mean
        G = len(cl_means)

        boot_means = np.empty(n_simulations)
        for b in range(n_simulations):
            # Rademacher weights: +1 or -1
            weights = np.random.choice([1, -1], size=G)
            pseudo = overall_mean + weights * resid.values
            boot_means[b] = pseudo.mean()

        lower = np.percentile(boot_means, 100 * (alpha / 2))
        upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

        results.append({
            'Metric': metric,
            'Mean': overall_mean,
            'Lower_CI': lower,
            'Upper_CI': upper,
            'CI_Width': upper - lower,
            'Confidence_Level': confidence_level
        })

    result_df = pd.DataFrame(results)

    if save_to:
        result_df.to_csv(save_to, index=False)

    return result_df

# wild_cluster_bootstrap_metrics(
#     metrics_df=metrics_df,
#     n_simulations=10000,
#     balance_participants=False,
#     balance_datasets=True,
#     group_by_column='Noise_Condition',
#     save_to='pretrained_bootstrap_results.csv')
