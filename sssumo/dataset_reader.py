# %%

import os
from io import StringIO

from math import ceil 
from glob import glob
import pickle
import json
import matplotlib.pyplot as plt
# 3d scatter plot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy import ndimage, interpolate
# from data import OrganicDataset, collate_fn

# %%
# Fitts functions
def summarize_fitts_structure(data, indent=0):
    """
    Recursively summarize the structure of JSON with indentation to show hierarchy.

    Args:
    - data: The JSON data to summarize (dict or list).
    - indent: Current indentation level for printing.
    """
    # Check if the data is a dictionary

    if isinstance(data, dict):
        print(' ' * indent + f"Dictionary with {len(data)} items:")
        for key, value in data.items():
            if key == 'block':
                continue
            print(' ' * indent + f"Key: '{key}' -> Type: {type(value).__name__}")
            summarize_fitts_structure(value, indent + 4)
            if key == 'movement_data' or (key == 'cursor' and isinstance(value, list)):
                if key == 'movement_data':
                    x = [i['cursor_x'] for i in value][1:]
                    y = [i['cursor_y'] for i in value][1:]
                    time = [i['evt_time'] for i in value][1:]
                elif key == 'cursor':
                    print('cursor')
                    x = [i['x'] for i in value]
                    y = [i['y'] for i in value]
                    time = np.arange(0, len(x), 1)

                x = np.array(x)
                y = np.array(y)
                time = np.array(time)
                time = time - time[0]

                velocity = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / np.diff(time)
                # smooth the velocity
                velocity = np.convolve(velocity, np.ones(10) / 10, mode='same')

                total_time = time[-1] - time[0]

                plt.subplot(2, 1, 1)
                plt.scatter(x, y, c=time)
                plt.title(f'Total time: {total_time}')
                plt.subplot(2, 1, 2)
                plt.plot(time[:-1], velocity)
                plt.show()

    
    # Check if the data is a list
    elif isinstance(data, list):
        print(' ' * indent + f"List of {len(data)} items -> Item Type: {type(data[0]).__name__ if len(data) > 0 else 'Empty'}")
        if len(data) > 0:  # Only traverse the first item to avoid printing entire list
            summarize_fitts_structure(data[-50], indent + 4)

    
    # If it's a scalar value, print its type
    else:
        print(' ' * indent + f"Value Type: {type(data).__name__}, Example: {data}")

def read_and_summarize_json(file_path):
    """
    Read a JSON file and summarize its structure.

    Args:
    - file_path: Path to the JSON file to read and summarize.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
        summarize_fitts_structure(data)

# %%
# Fitts processing
def extract_cursor_data_from_file(json_file_path, start_index=0):

    with open(json_file_path, 'r') as file:
        data = json.load(file)
    file_name = json_file_path.split('/')[-1]

    participant = file_name.split('.')[0][-3:]

    cursor_df_list = []
    trials = data['trials']
    trials_frames = trials['frames']
    for i, trial in enumerate(trials_frames):
        cursor_df = pd.DataFrame(trial['cursor'])
        cursor_df['trial_n'] = i + start_index
        # print(cursor_df.head())
        # print(trial)
        trial['vbl_time'] = np.array(trial['vbl_time'])
        cursor_df['time'] = trial['vbl_time'] - trial['vbl_time'][0]
        cursor_df['disp_time'] = trial['disp_time']

        cursor_df_list.append(cursor_df)
    cursor_df_all = pd.concat(cursor_df_list)
    cursor_df_all.reset_index(drop=True, inplace=True)
    cursor_df_all['file_name'] = file_name
    cursor_df_all['participant'] = participant

    return cursor_df_all

def fitts_to_pandas(json_file_pattern, pandas_path):
    json_file_paths = glob(json_file_pattern)
    json_file_paths.sort()
    cursor_df_list = []
    number_trials = 0
    for file_path in json_file_paths:
        print(f'Processing {file_path}...')
        cursor_df = extract_cursor_data_from_file(file_path, start_index=number_trials)
        cursor_df_list.append(cursor_df)
        number_trials += cursor_df['trial_n'].nunique()
    cursor_df_all = pd.concat(cursor_df_list)
    cursor_df_all.reset_index(drop=True, inplace=True)
    cursor_df_all.to_csv(pandas_path, index=False)

    print(f'Saved cursor data to {pandas_path}')


# %%
# Object moving preprocessing (Bochum)
def object_moving_to_pandas(dat_file_path, pandas_path):

    repetitions = 10
    participants = 10
    experiments = 16
    num_dofs = 3
    sampling_rate = 110

    dat_df = pd.read_csv(dat_file_path, sep=' ')
    dat_values = dat_df.values
    dat_values = dat_values.reshape(-1, num_dofs, repetitions, participants, experiments)

    data_df_list = []

    for participant, experiment, repetition in np.ndindex(participants, experiments, repetitions):

        data = dat_values[:, :, repetition, participant, experiment]

        # fill nan data with linear interpolation
        timestamps = np.arange(0, len(data)) / sampling_rate
        last_nonnan_timestamp = 0
        for i in range(num_dofs):
            nan_mask = np.isnan(data[:, i])
            nan_timestamps = timestamps[nan_mask]
            nonnan_timestamps = timestamps[~nan_mask]
            nonnan_signal = data[~nan_mask, i]
            data[nan_mask, i] = np.interp(nan_timestamps, nonnan_timestamps, nonnan_signal)

            last_nonnan_timestamp = max(last_nonnan_timestamp, nonnan_timestamps[-1])

        data = data[timestamps <= last_nonnan_timestamp]
        timestamps = timestamps[timestamps <= last_nonnan_timestamp]

        # most_nan = max(np.isnan(data).sum(axis=0))
        # if most_nan > 0:
        #     data = data[:-most_nan]

        data_df = pd.DataFrame(data, columns=['x', 'y', 'z'])
        data_df['repetition'] = repetition
        data_df['participant'] = participant
        data_df['experiment'] = experiment
        data_df['trial_n'] = participant * experiments * repetitions + experiment * repetitions + repetition
        data_df['time'] = timestamps
        data_df_list.append(data_df)

    data_df_all = pd.concat(data_df_list)
    data_df_all.reset_index(drop=True, inplace=True)
    data_df_all.to_csv(pandas_path, index=False)



def get_rotation_matrix_batch(yaw_pitch_roll_angles):
    """
    Returns a batch of rotation matrices given a Nx3 matrix of yaw, pitch, and roll angles.
    """
    yaw = np.radians(yaw_pitch_roll_angles[:, 0])
    pitch = np.radians(yaw_pitch_roll_angles[:, 1])
    roll = np.radians(yaw_pitch_roll_angles[:, 2])
    
    sinalpha = np.sin(yaw)
    cosalpha = np.cos(yaw)
    sinbeta = np.sin(pitch)
    cosbeta = np.cos(pitch)
    singamma = np.sin(roll)
    cosgamma = np.cos(roll)

    rotation_matrices = np.zeros((yaw_pitch_roll_angles.shape[0], 3, 3))
    
    rotation_matrices[:, 0, 0] = cosalpha * cosbeta
    rotation_matrices[:, 0, 1] = cosalpha * sinbeta * singamma - sinalpha * cosgamma
    rotation_matrices[:, 0, 2] = cosalpha * sinbeta * cosgamma + sinalpha * singamma

    rotation_matrices[:, 1, 0] = sinalpha * cosbeta
    rotation_matrices[:, 1, 1] = sinalpha * sinbeta * singamma + cosalpha * cosgamma
    rotation_matrices[:, 1, 2] = sinalpha * sinbeta * cosgamma - cosalpha * singamma

    rotation_matrices[:, 2, 0] = -sinbeta
    rotation_matrices[:, 2, 1] = cosbeta * singamma
    rotation_matrices[:, 2, 2] = cosbeta * cosgamma

    return rotation_matrices

def rotate_vector(yaw_pitch_roll_angles, vector_to_rotate=None):
    """
    Rotates a batch of vectors by a batch of yaw, pitch, and roll angles.
    yaw_pitch_roll_angles: Nx3 matrix where each row is [yaw, pitch, roll] angles.
    vectors: Nx3 matrix of vectors to rotate, or defaults to [0, 0, 1] for each row.
    """
    num_vectors = yaw_pitch_roll_angles.shape[0]
    
    if vector_to_rotate is None:
        vector_to_rotate = get_unit_vector()
        # Default to unit vectors along the z-axis
    vectors = np.tile(vector_to_rotate, (num_vectors, 1))
    vectors = vectors[..., None]  # Add a new axis for matrix multiplication
    
    # Get the batch of rotation matrices
    rotation_matrices = get_rotation_matrix_batch(yaw_pitch_roll_angles)
    
    # Perform batch matrix multiplication (N x 3 x 3) @ (N x 3 x 1) -> (N x 3)
    rotated_vectors = np.matmul(rotation_matrices, vectors)
    rotated_vectors = rotated_vectors.squeeze(-1)
    
    return rotated_vectors

def get_unit_vector():
    """
    Returns a unit vector along the z-axis.
    """
    return np.array([0, 0, 1])

# Example usage:
# yaw_pitch_roll_angles = np.array([[30, 45, 60], [90, 0, 30], [180, 90, 0]])  # Nx3 matrix
# rotated_vectors = rotate_vectors(yaw_pitch_roll_angles)
# print(rotated_vectors)

def output_2d_scatter(xs, ys, path=None):
    """
    Draw a 2D scatter plot given data points and output to path.
    """
    plt.figure()
    plt.scatter(xs, ys, c=[i for i in range(xs.shape[0])], cmap='hot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

def output_3d_scatter(xs, ys, zs, path=None):
    """
    Draw a 3D scatter plot given data points and output to path.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(xs, ys, zs, c=[i for i in range(xs.shape[0])], cmap='hot')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

def stanford_handwriting_data_to_pandas(csv_files_pattern, header_file_path, pandas_path):
    csv_files_paths = glob(csv_files_pattern)
    csv_files_paths.sort()

    header_df = pd.read_csv(header_file_path, skipinitialspace=True)
    columns = header_df.columns.tolist()

    data_df_list = []

    num_trials = 0

    for i, csv_file_path in enumerate(csv_files_paths):
        letter = csv_file_path.split('/')[-1].split('.')[0]
        if letter == 'calibration':
            continue
    # print(f'Processing {csv_file_path}...')
        data_df = pd.read_csv(csv_file_path, names=columns)
        sample_index2order = {sample_index: i + num_trials for i, sample_index in enumerate(data_df['sample_index'].unique())}
        data_df['trial_n'] = data_df['sample_index'].map(sample_index2order)
        num_trials += len(sample_index2order)



        yaw_pitch_roll_angles = data_df[['yaw', 'pitch', 'roll']].values
        xyz_vec = rotate_vector(yaw_pitch_roll_angles)
        data_df['x'] = xyz_vec[:, 0]
        data_df['y'] = xyz_vec[:, 1]
        data_df['z'] = xyz_vec[:, 2]
        data_df['time_delta'] = data_df['time_delta'].astype(float)
        data_df['time'] = data_df['time_delta'].groupby(data_df['sample_index']).transform('cumsum')
        data_df['time'] /= 1000  # Convert to seconds


        file_name_parts = csv_file_path.split('/')
        data_df['subject_id'] = file_name_parts[-2]
        data_df['letter'] = letter
        
        data_df_list.append(data_df)

    data_df_all = pd.concat(data_df_list)

    data_df_all.reset_index(drop=True, inplace=True)
    data_df_all.to_csv(pandas_path, index=False)

# %%
# Whac-a-mole preprocessing (Continuous Mouse Tracking)
class MouseDataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Remap Int64Index to the appropriate location
        if name == 'Int64Index':
            return pd.Index  # Or pd.RangeIndex if applicable
        return super().find_class(module, name)

def whacamole_to_pandas(pickle_files_pattern, pandas_path):
    pickle_files = glob(pickle_files_pattern)
    
    trial_n = 0


    data_df_list = []

    for _, pickle_file in enumerate(pickle_files):
        print(f'Processing {pickle_file}...')
        with open(pickle_file, 'rb') as f:
            df = MouseDataUnpickler(f).load()

            # print(df.head())
            # print(df.columns)


        for i in range(len(df)):
            x_cord = df.iloc[i].Xcord
            y_cord = df.iloc[i].Ycord

            timestamps = df.iloc[i].Timemousecords
        
            data_df = pd.DataFrame({'x': x_cord, 'y': y_cord, 'time': timestamps})
            data_df['trial_n'] = trial_n
            data_df['participant'] = df.iloc[i].ID
            data_df['device'] = df.iloc[i].device
            data_df['file_name'] = pickle_file.split('/')[-1]
            data_df['time'] /= 1000  # Convert to seconds
            data_df['time'] -= data_df['time'].min()
            data_df_list.append(data_df)
            trial_n += 1

    data_df_all = pd.concat(data_df_list)
    data_df_all.reset_index(drop=True, inplace=True)
    data_df_all.to_csv(pandas_path, index=False)

# %%

vizualize_whacamole = False

if vizualize_whacamole:
    whacamole_pickle = '~/data/human_movement/2d.2.whac-a-mole/game_data_300_feb_O.p'
    # load pickle
    with open(whacamole_pickle, 'rb') as f:
        df = MouseDataUnpickler(f).load()

    row_n = 54

    x_cord = df.iloc[row_n].Xcord
    y_cord = df.iloc[row_n].Ycord

    timestamps = df.iloc[row_n].Timemousecords

    cords = np.array([x_cord, y_cord]).T

    x_clicks = df.iloc[row_n]['mouseCordsXClicks']
    y_clicks = df.iloc[row_n]['mouseCordsYClicks']

    timestamps_clicks = df.iloc[row_n]['clicktimes']

    clicks = np.array([x_clicks, y_clicks]).T

    
    # scatter clicks
    # plt.figure(dpi=150)
    # plt.scatter(clicks[:, 0], clicks[:, 1], c=[i for i in range(len(clicks))], cmap='hot')



    # plt.plot(cords[:, 0], cords[:, 1])
    # plt.show()

    timestamps = np.array(timestamps)
    timestamps -= timestamps[0]
    timestamps /= 1000

    time_start = 12.75
    time_end = 14.5

    blue_plotted = False

    alpha = 0.2
    color = '#606060'

    timestamp_mask = (timestamps >= time_start) * ( timestamps <= time_end)

    plt.figure(figsize=(10, 10), dpi=300)
    for i in range(len(cords) - 1):
        if timestamps[i] >= time_start and timestamps[i+1] <= time_end:
            if blue_plotted:
                continue
            plt.plot(cords[timestamp_mask, 0], cords[timestamp_mask, 1], color='blue', alpha=1, linewidth=5)
            blue_plotted = True

        plt.plot(cords[i:i+2, 0], cords[i:i+2, 1], color=color, alpha=alpha, linewidth=3, solid_capstyle="butt")

    # plt.plot(clicks[:, 0], clicks[:, 1], color='blue', linewidth=5)
    # plot a hoolow circle at 0, 0
    plt.scatter(357, 520, color='red', marker='o', s=600, alpha=0.8, facecolors='none', edgecolors='r', linewidth=2.5)
    # add a text '1'
    plt.text(370, 520, '1', fontsize=25, color='black')

    plt.scatter(610, 310, color='red', marker='o', s=600, alpha=0.8, facecolors='none', edgecolors='r', linewidth=2.5)
    # add a text '2'
    plt.text(623, 310, '2', fontsize=25, color='black')

    # set aspect ratio to be equal
    plt.xlim(175, 725)
    plt.ylim(100, 650)
    plt.gca().set_aspect('equal', adjustable='box')

    # remove axes
    plt.axis('off')

    plt.savefig('mouse_movement.png', dpi=300)

    plt.show()

# %%
vizualize_fitts = False

if vizualize_fitts:
    fitts_path = '/Users/NAME_REMOVED/data/human_movement/2d.1.Fitts_task/Fitts_data.csv'
    fitts_df = pd.read_csv(fitts_path)

    fitts_df_participant = fitts_df[fitts_df['file_name'] == 'GainAdaptSub001.json']



    fitts_df_trial = fitts_df[fitts_df['trial_n'] == 405]
    fitts_df_trial['time'] -= fitts_df_trial['time'].min()

    time_mask = (fitts_df_trial['time'] > 1.5) & (fitts_df_trial['time'] <= 25)
    fitts_df_trial = fitts_df_trial[time_mask]

    plt.figure(figsize=(10, 10), dpi=300)

    # for i in range(len(fitts_df_participant) - 1):
    #     plt.plot(fitts_df_participant.iloc[i:i+2]['x'],
    #              fitts_df_participant.iloc[i:i+2]['y'],
    #              color='#606060',
    #              alpha=0.2,
    #              linewidth=3, solid_capstyle="butt")

    unique_trials = fitts_df_participant['trial_n'].unique()
    for trial_n in unique_trials:
        fitts_df_trial_gray = fitts_df_participant[fitts_df_participant['trial_n'] == trial_n]
        plt.plot(fitts_df_trial_gray['x'], fitts_df_trial_gray['y'], color='#606060', alpha=0.075, linewidth=3, solid_capstyle="butt")

    # plt.scatter(fitts_df_participant['x'], fitts_df_participant['y'], alpha=0.1
    plt.plot(fitts_df_trial['x'], fitts_df_trial['y'], color='blue', linewidth=5)
    plt.scatter(0, -52, color='red', marker='o', s=600, alpha=0.8, facecolors='none', edgecolors='r', linewidth=2.5)


    # set aspect ratio to be equal
    plt.xlim(-125, 125)
    plt.ylim(-200, 50)
    plt.gca().set_aspect('equal', adjustable='box')

    # remove axes
    plt.axis('off')

    plt.savefig('fitts_task.png', dpi=300)


    plt.show()




# plt.plot(fitts_df_trial['time'], fitts_df_trial['y'])
# plt.show()

# %%
# Steering preprocessing (CogCarSim)
def steering_to_pandas(steps_path, runs_path, pandas_path):
    original_df = pd.read_csv(steps_path)
    runs_df = pd.read_csv(runs_path)
    unique_runs = runs_df['run'].unique()
    run2trial_n = {run: int(i) for i, run in enumerate(unique_runs)}
    run2participant = {run: int(runs_df.loc[runs_df['run'] == run, 'participant_n'].values[0]) for run in unique_runs}
    new_df = pd.DataFrame(
        {'x': original_df['wheelpos'],
         'time': original_df['clock_begin'],
         'run': original_df['run'],
         'trial_n': original_df['run'].map(run2trial_n),
         'participant': original_df['run'].map(run2participant)}
    )
    new_df['participant'] = new_df['participant'].fillna(-1).astype(int)
    new_df['trial_n'] = new_df['trial_n'].fillna(-1).astype(int)
    new_df.to_csv(pandas_path, index=False)


# %%
# Pointing preprocessing
def pointing_to_pandas(csv_paths_pattern, pandas_path):
    csv_paths = glob(csv_paths_pattern)
    csv_paths.sort()

    original_columns = ['trial_id', 
                        'target_x', 'target_y', 'target_z', 
                        'x_filtered', 'y_filtered', 'z_filtered', 
                        'time_filtered', 
                        'x', 'y', 'z', 
                        'time', 'empty'
                        ]

    data_df_list = []
    last_trial_n = 0
    for csv_path in csv_paths:
        participant_letter = csv_path.split('/')[-1].split('.')[0][-1]

        data_df = pd.read_csv(csv_path, names=original_columns, skiprows=1, index_col=None)
        del data_df['empty']

        data_df['trial_n'] = data_df['trial_id'].astype(int) + last_trial_n
        data_df['time'] = data_df.groupby('trial_id')['time'].transform(lambda x: x - x.min())
        last_trial_n = data_df['trial_n'].max()

        data_df['participant'] = participant_letter
        data_df_list.append(data_df)

    data_df_all = pd.concat(data_df_list)
    data_df_all.dropna(inplace=True)
    data_df_all.reset_index(drop=True, inplace=True)
    data_df_all.to_csv(pandas_path, index=False)


# %%
# Tablet writing preprocessing
def tablet_writing_to_pandas(dir_paths_patterns: dict, pandas_path: str):
    data_df_list = []
    trial_n = 0

    original_columns = ['time', 'x', 'y', 'z', 'pressure']
    for dataset_group, dir_path_pattern in dir_paths_patterns.items():
        dir_paths = glob(dir_path_pattern)
        dir_paths.sort()
        for dir_path in dir_paths:
            recording_id = dir_path.split('/')[-1].split('Recording')[1]

            touch_data_path = os.path.join(dir_path, 'stylus_touch_data.csv')
            hover_data_path = os.path.join(dir_path, 'stylus_hover_data.csv')
            meta_path = os.path.join(dir_path, 'meta.json')
            labels_path = os.path.join(dir_path, 'labels.csv')


            touch_df = pd.read_csv(touch_data_path, sep=';', names=original_columns, skiprows=1, dtype=float)
            hover_df = pd.read_csv(hover_data_path, sep=';', names=original_columns, skiprows=1, dtype=float)
            meta = json.load(open(meta_path))
            labels_df = pd.read_csv(labels_path, sep=';')

            participant = meta['person_name_or_id']

            # print(participant)


            position_df = pd.concat([touch_df, hover_df])

            # for col in original_columns:
            #     position_df[col] = position_df[col].astype(float)

            #position_df['time'] = position_df['time'].astype(float)

            position_df['participant'] = participant
            position_df['recording_id'] = recording_id
            position_df['trial_n'] = -1
            position_df['dataset_group'] = dataset_group

            position_df['label'] = None

            for _, row in labels_df.iterrows():
                label, start, end = row.values[:3]
                position_df.loc[(position_df['time'] >= start) & (position_df['time'] <= end), 'label'] = label
                position_df.loc[(position_df['time'] >= start) & (position_df['time'] <= end), 'trial_n'] = trial_n
                trial_n += 1


            position_df['time'] -= position_df['time'].min()
            position_df['time'] /= 1000
            position_df.sort_values(by='time', inplace=True)
            position_df.reset_index(drop=True, inplace=True)

            data_df_list.append(position_df)

    data_df_all = pd.concat(data_df_list)
    data_df_all.reset_index(drop=True, inplace=True)
    data_df_all.to_csv(pandas_path, index=False)

# %%
# Crank preprocessing
def crank_to_pandas(asc_paths_pattern, pandas_path):
    sr = 200
    threshold = 5

    condition_mapping = [
        'no_feedback',
        'slow_clock',
        'slow_anticlock',
        'medium_clock',
        'medium_anticlock',
        'fast_clock',
        'fast_anticlock',
        ]

    asc_paths = glob(asc_paths_pattern)
    asc_paths.sort()

    original_columns = ['x', 'velocity', 'force_X', 'force_Y', 'force_Z', 'moment_X', 'moment_Y', 'moment_Z', 'emg_bicep', 'emg_tricep']

    data_df_list = []

    trial_n = 0

    for asc_path in asc_paths:

        participant = asc_path.split('/')[-2]
        recording_id = asc_path.split('/')[-1].split('.')[0]
        condition_id = int(recording_id[3])
        condition = condition_mapping[condition_id]

        # strip all lines in asc_path
        with open(asc_path, 'r') as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        # print(lines)
        csv_str = '\n'.join(lines)
        # print(csv_str)

        asc_df = pd.read_csv(StringIO(csv_str), sep=' ', names=original_columns)

        asc_df['participant'] = participant
        asc_df['recording_id'] = recording_id
        asc_df['condition'] = condition

        x = asc_df['x'].values
        x_diff = np.diff(x, prepend=x[0])
        for i in range(len(x_diff)):
            if x_diff[i] > threshold:
                x[i:] -= 2 * np.pi
            elif x_diff[i] < -threshold:
                x[i:] += 2 * np.pi
        
        asc_df['x'] = x
        asc_df['trial_n'] = trial_n
        trial_n += 1

        asc_df['time'] = np.arange(0, len(asc_df), 1) / sr

        data_df_list.append(asc_df)

    data_df_all = pd.concat(data_df_list)
    data_df_all.reset_index(drop=True, inplace=True)
    data_df_all.to_csv(pandas_path, index=False)

# %%
# Directional tangential velocity functions

def directional_angle_3d(vectors, smoothing_kernel=10):
    # Calculate the vectors between consecutive points
    v1 = vectors[:-1]  # All vectors except the last one
    v2 = vectors[1:]   # All vectors except the first one
    
    # Calculate the dot product between consecutive vectors
    dot_products = np.sum(v1 * v2, axis=1)
    
    # Calculate the norms (magnitudes) of the consecutive vectors
    norms_v1 = np.linalg.norm(v1, axis=1)
    norms_v2 = np.linalg.norm(v2, axis=1)
    
    # Calculate the cosine of the angle between consecutive vectors
    cos_theta = dot_products / (norms_v1 * norms_v2)
    
    # Calculate the angles in radians and convert to degrees
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta_degrees = np.degrees(theta)
    
    # Calculate the cross products between consecutive vectors
    cross_products = np.cross(v1, v2)
    
    # Use the z-component (or another reference axis) to determine direction
    z_components = cross_products[:, 2]  # Extract the z-component
    
    # If the z-component is negative, the angle is clockwise (negative)
    theta_degrees[z_components < 0] *= -1
    
    return theta_degrees

def vectors2direction_change(vectors, smoothing_kernel=3, step=None, angle_threshold=90, return_angle=False):


    if step is None:
        step = smoothing_kernel
    # pad vectors with zeros
    # assert smoothing_kernel % 2 == 1, 'Smoothing kernel must be odd'
    vectors = np.concatenate([np.ones((step//2, vectors.shape[1])) * vectors[0],
                              vectors,
                              np.ones((ceil(step /2) - 1, vectors.shape[1])) * vectors[-1]])

    vectors = ndimage.uniform_filter1d(vectors, smoothing_kernel, axis=0)
    cumulative_vectors = np.cumsum(vectors, axis=0)
    # plt.plot(cumulative_vectors[:, 0], cumulative_vectors[:, 1])
    # plt.show()

    dot_product = np.sum(vectors[step:] * vectors[:-step], axis=1)
    magnitude = np.linalg.norm(vectors, axis=1)

    mean_magnitude = np.mean(magnitude)

    # for i, magn in enumerate(magnitude):
    #     print(f'{i}: {magn}')


    cos = dot_product / (magnitude[step:] * magnitude[:-step])
    # plt.plot(np.log(magnitude))
    # plt.show()
    angle = np.arccos(cos)
    angle = np.degrees(angle)
    angle[np.isnan(angle)] = 0
    angle[magnitude[step:] < mean_magnitude * 1e-5] = 0
    angle[magnitude[:-step] < mean_magnitude * 1e-5] = 0

    angle = np.concatenate([[0], angle])
    peaks_mask = (angle[1:-1] >= angle[:-2]) & (angle[1:-1] > angle[2:])
    peaks_mask = np.concatenate([[False], peaks_mask, [False]])
    direction_changes = np.zeros_like(angle)
    direction_changes[peaks_mask & (angle > angle_threshold)] = 1

    
    if return_angle:
        return direction_changes, angle
    
    return direction_changes

def coordinates_to_tangential_velocity(x, y=None, z=None, timestamps=None, sampling_rate=None, target_sampling_rate=None, directional=True, debug=False,
                                       direction_smoothing_kernel=5, direction_step=None,
                                       angle_threshold=90, return_x_y_z_time=False):

    assert (timestamps is not None
            or target_sampling_rate is not None 
            or sampling_rate is not None)
    if isinstance(x, pd.DataFrame):
        timestamps = x['time'].values
        y = x['y'].values if 'y' in x.columns else None
        z = x['z'].values if 'z' in x.columns else None
        x = x['x'].values
        
    if y is None:
        y = np.zeros_like(x)
    if z is None:
        z = np.zeros_like(x)

    if (target_sampling_rate is None) and (sampling_rate is not None):
        target_sampling_rate = sampling_rate
    if timestamps is None:
        if sampling_rate is None:
            print(f'Note! No time and sampling rate provided, assuming {target_sampling_rate} Hz'
                  'If this is not right, the results may be incorrect.')
            sampling_rate = target_sampling_rate
        timestamps = np.arange(0, len(x), 1) / sampling_rate
    if sampling_rate is None:
        sampling_rate = 1 / np.mean(np.diff(timestamps))

    timestamps -= timestamps[0] # Ensure that time starts from 0

    sampling_rate_ratio = target_sampling_rate / sampling_rate
    sampling_rate_ratio = ceil(sampling_rate_ratio)
    x = ndimage.uniform_filter1d(x, sampling_rate_ratio)
    y = ndimage.uniform_filter1d(y, sampling_rate_ratio)
    z = ndimage.uniform_filter1d(z, sampling_rate_ratio)


    interpolated_timestamps = np.arange(0, timestamps[-1], 1 / target_sampling_rate)
    
    x = interpolate.interp1d(timestamps, x, kind='linear')(interpolated_timestamps)
    y = interpolate.interp1d(timestamps, y, kind='linear')(interpolated_timestamps)
    z = interpolate.interp1d(timestamps, z, kind='linear')(interpolated_timestamps)

    x_diff = np.diff(x, prepend=x[0])
    y_diff = np.diff(y, prepend=y[0])
    z_diff = np.diff(z, prepend=z[0])
    time_diff = np.diff(interpolated_timestamps)
    time_diff = np.concatenate([[time_diff[0]], time_diff])

    directions = np.array([x_diff, y_diff, z_diff]).T

    tangential_velocity = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

    if directional:
        direction_changes, angles = vectors2direction_change(directions, smoothing_kernel=direction_smoothing_kernel, step=direction_step, angle_threshold=angle_threshold, return_angle=True)
        direction_changes_cumsum = np.cumsum(direction_changes)
        direction_signs = np.where(direction_changes_cumsum % 2 == 0, 1, -1)
        tangential_velocity = tangential_velocity * direction_signs


    if debug:

        plt.figure(dpi=150)
        plt.plot(interpolated_timestamps, angles)
        plt.scatter(interpolated_timestamps[direction_changes == 1], angles[direction_changes == 1], c='red', marker='x')
        plt.xlabel('Time')
        plt.ylabel('Angle')
        plt.title('Directional changes')
        plt.show()


        plt.figure(dpi=150)
        plt.plot(interpolated_timestamps, tangential_velocity)
        if directional:
            plt.scatter(interpolated_timestamps[direction_changes == 1], tangential_velocity[direction_changes == 1], c='red', marker='x')
        plt.xlabel('Time (s)')
        plt.ylabel('Tangential velocity')
        plt.title('Calculated tangential velocity')
        plt.show()

        
        if np.abs(y).sum() > 0:
            plt.figure(dpi=150)
            if np.abs(z).sum() > 0:
                plt.scatter(x, y, z, c=np.abs(tangential_velocity), label='Movement with abs. tangential velocity')
                if directional:
                    plt.scatter(x[direction_changes == 1], y[direction_changes == 1], z[direction_changes == 1], c='red', marker='x', label='Directional change')
                plt.scatter(x[0], y[0], z[0], c='green', marker='x', label='Start')
                plt.scatter(x[-1], y[-1], z[-1], c='green', marker='^', label='End')
                # plt.zlabel('Z')
            else:
                plt.scatter(x, y, c=np.abs(tangential_velocity), label='Movement with abs. tangential velocity')
                if directional:
                    plt.scatter(x[direction_changes == 1], y[direction_changes == 1], c='red', marker='x', label='Directional change')
                plt.scatter(x[0], y[0], c='green', marker='x', label='Start')
                plt.scatter(x[-1], y[-1], c='green', marker='^', label='End')

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Movement with directional changes')
            plt.legend()
            plt.show()


    if return_x_y_z_time:
        return tangential_velocity, (x, y, z,  interpolated_timestamps)
    return tangential_velocity



def df_to_tangential_velocity(original_df_path, updated_df_path, 
                            timestamps=None, sampling_rate=None, target_sampling_rate=None, directional=True, debug=False,
                            direction_smoothing_kernel=5, direction_step=None, angle_threshold=90, x_col='x', y_col='y', z_col='z', time_col='time', trial_col=['trial_n'], trial_n_range=None):
    
    # disable RuntimeWarning
    np.seterr(divide='ignore', invalid='ignore')
    # iterate over trials
    df = pd.read_csv(original_df_path)
    if trial_n_range is not None:
        df = df[(df[trial_col[0]] >= trial_n_range[0]) * (df[trial_col[0]] <= trial_n_range[1])]
    trial_df_list_with_tangential_velocity = []
    for trial_n, trial_df in df.groupby(trial_col):
        if trial_n[0] == -1:
            continue
        if len(trial_df) < 10:
            print(f'Skipping trial {trial_n[0]} with less than 10 samples')
            continue
        x = trial_df[x_col].values


        y = trial_df[y_col].values if y_col in trial_df.columns else None
        z = trial_df[z_col].values if z_col in trial_df.columns else None
        timestamps = trial_df[time_col].values

        tangential_velocity, (x, y, z, timestamps) = coordinates_to_tangential_velocity(x, y=y, z=z, timestamps=timestamps, sampling_rate=sampling_rate, target_sampling_rate=target_sampling_rate, directional=directional, debug=debug, return_x_y_z_time=True, direction_smoothing_kernel=direction_smoothing_kernel, angle_threshold=angle_threshold,
                                                                                        direction_step=direction_step)
        trial_df_with_tangential_velocity = pd.DataFrame({'tangential_velocity': tangential_velocity, 'x': x, 'y': y, 'z': z, 'time': timestamps})
        trial_df_with_tangential_velocity['trial_n'] = trial_n[0]
        for col in trial_df.columns:
            if col not in trial_df_with_tangential_velocity.columns:
                trial_df_with_tangential_velocity[col] = trial_df[col].values[0]
        trial_df_list_with_tangential_velocity.append(trial_df_with_tangential_velocity)

    df_with_tangential_velocity = pd.concat(trial_df_list_with_tangential_velocity)
    df_with_tangential_velocity.reset_index(drop=True, inplace=True)
    df_with_tangential_velocity.to_csv(updated_df_path, index=False)


# %%
# Preprocess all datasets to the same dataframe format
preprocess_data = True

if preprocess_data:
    # Crank preprocessing
    crank_asc_pattern = '/Users/NAME_REMOVED/data/human_movement/1d.2.crank/*/*.ASC'
    crank_pandas_path = '/Users/NAME_REMOVED/data/human_movement/1d.2.crank/crank_data.csv'
    crank_tangential_velocity_pandas_path = '/Users/NAME_REMOVED/data/human_movement/crank_tangential_velocity_data.csv'
    crank_to_pandas(crank_asc_pattern, crank_pandas_path)
    df_to_tangential_velocity(crank_pandas_path, crank_tangential_velocity_pandas_path)

    # Tablet writing preprocessing
    tablet_writing_dir_paths_patterns = {
        'IRISA-KIHT-S': '/Users/NAME_REMOVED/data/human_movement/3d.3.writing/IRISA-KIHT-S-Dataset/IRISA-KIHT-S/*',
    'KIHT-Public-Flat': '/Users/NAME_REMOVED/data/human_movement/3d.3.writing/KIHT-Public/Recording/Flat/*',
        'KIHT-Public-Inclined': '/Users/NAME_REMOVED/data/human_movement/3d.3.writing/KIHT-Public/Recording/Inclined/*',
    }
    tablet_writing_tangential_velocity_pandas_path = '/Users/NAME_REMOVED/data/human_movement/3d.3.writing/tablet_writing_tangential_velocity_data.csv'
    tablet_writing_pandas_path = '/Users/NAME_REMOVED/data/human_movement/tablet_writing_data.csv'
    tablet_writing_to_pandas(tablet_writing_dir_paths_patterns, tablet_writing_pandas_path)
    df_to_tangential_velocity(tablet_writing_pandas_path, tablet_writing_tangential_velocity_pandas_path)

    # Pointing preprocessing
    pointing_csv_paths_pattern = '/Users/NAME_REMOVED/data/human_movement/3d.2.pointing/DataS1/*.csv'
    pointing_pandas_path = '/Users/NAME_REMOVED/data/human_movement/3d.2.pointing/pointing_data.csv'
    pointing_tangential_velocity_pandas_path = '/Users/NAME_REMOVED/data/human_movement/pointing_tangential_velocity_data.csv'
    pointing_to_pandas(pointing_csv_paths_pattern, pointing_pandas_path)
    df_to_tangential_velocity(pointing_pandas_path, pointing_tangential_velocity_pandas_path)
    
    # Steering preprocessing
    steps_df_path = '/Users/NAME_REMOVED/data/cogcarsim/intermediate/step_df.csv'
    runs_df_path = '/Users/NAME_REMOVED/data/cogcarsim/intermediate/run_df.csv'
    steering_pandas_path = '/Users/NAME_REMOVED/data/human_movement/1d.1.steering/steering_data.csv'
    steering_tangential_velocity_pandas_path = '/Users/NAME_REMOVED/data/human_movement/steering_tangential_velocity_data.csv'
    steering_to_pandas(steps_df_path, runs_df_path, steering_pandas_path)
    df_to_tangential_velocity(steering_pandas_path, steering_tangential_velocity_pandas_path)

    # Object moving preprocessing
    object_moving_dat_path = '/Users/NAME_REMOVED/data/human_movement/3d.1.object_moving/armTrajectories.dat'
    object_moving_pandas_path = '/Users/NAME_REMOVED/data/human_movement/3d.1.object_moving/object_moving_data.csv'
    object_moving_tangential_velocity_pandas_path = '/Users/NAME_REMOVED/data/human_movement/object_moving_tangential_velocity_data.csv'
    object_moving_to_pandas(object_moving_dat_path, object_moving_pandas_path)
    df_to_tangential_velocity(object_moving_pandas_path, object_moving_tangential_velocity_pandas_path)

    # Fitts preprocessing
    fitts_json_paths_pattern = '/Users/NAME_REMOVED/data/human_movement/2d.1.Fitts_task/inlab/GainAdaptSub*.json'
    fitts_pandas_path = '/Users/NAME_REMOVED/data/human_movement/2d.1.Fitts_task/Fitts_data.csv'
    fitts_tangential_velocity_pandas_path = '/Users/NAME_REMOVED/data/human_movement/Fitts_tangential_velocity_data.csv'
    fitts_to_pandas(fitts_json_paths_pattern, fitts_pandas_path)
    df_to_tangential_velocity(fitts_pandas_path, fitts_tangential_velocity_pandas_path)

    # Whac-a-mole preprocessing
    whacamole_pickle_files_pattern = '/Users/NAME_REMOVED/data/human_movement/2d.2.whac-a-mole/*.p'
    whacamole_pandas_path = '/Users/NAME_REMOVED/data/human_movement/2d.2.whac-a-mole/whacamole_data.csv'
    whacamole_tangential_velocity_pandas_path = '/Users/NAME_REMOVED/data/human_movement/whacamole_tangential_velocity_data.csv'
    whacamole_to_pandas(whacamole_pickle_files_pattern, whacamole_pandas_path)
    df_to_tangential_velocity(whacamole_pandas_path, whacamole_tangential_velocity_pandas_path)

# %%
# Tablet writing preprocessing

tablet_writing_tangential_velocity_pandas_path = '/Users/NAME_REMOVED/data/human_movement/tablet_writing_tangential_velocity_data_high_sampling_rate3.csv'
tablet_writing_pandas_path = '/Users/NAME_REMOVED/data/human_movement/3d.3.writing/tablet_writing_data.csv'
df_to_tangential_velocity(tablet_writing_pandas_path, tablet_writing_tangential_velocity_pandas_path,
                          target_sampling_rate=370,
                          direction_smoothing_kernel=3, 
                          direction_step=3)

# %%
def plot_random_trials(df_paths, n_trials=10):
    for df_path in df_paths:
        df = pd.read_csv(df_path)
        trial_numbers = np.random.choice(df['trial_n'].unique(), n_trials)
        for trial_number in trial_numbers:
            df_trial = df[df['trial_n'] == trial_number]
            timestamps = df_trial['time'].values
            tangential_velocity = df_trial['tangential_velocity'].values
            plt.plot(timestamps[:600], tangential_velocity[:600])
            plt.title(f'{df_path.split("/")[-1]} Trial {trial_number}')
            plt.xlabel('Time')
            plt.ylabel('Tangential velocity')
            plt.show()

# %%
plot_random_trials([crank_tangential_velocity_pandas_path, tablet_writing_tangential_velocity_pandas_path, pointing_tangential_velocity_pandas_path, steering_tangential_velocity_pandas_path, object_moving_tangential_velocity_pandas_path, fitts_tangential_velocity_pandas_path, whacamole_tangential_velocity_pandas_path])

# %%
crank_data_df = pd.read_csv(crank_pandas_path)
crank_data_df.head()


# %%
x = crank_data_df['x'].values[:1400]
x_diff = np.diff(x, prepend=x[0])
velocity = crank_data_df['velocity'].values[:1400]
x_diff / velocity
# %%
pointing_to_pandas(pointing_csv_paths_pattern, pointing_pandas_path)
df_to_tangential_velocity(pointing_pandas_path, pointing_tangential_velocity_pandas_path)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%

tablet_writing_pandas_path = '/Users/NAME_REMOVED/data/human_movement/3d.3.writing/tablet_writing_data.csv'
tablet_writing_df = pd.read_csv(tablet_writing_pandas_path)

# %%
def detect_segments(df, criterions=('z', 'tangential_velocity')):
    # Sort by index to ensure order
    df = df.sort_index()
    
    # Create a mask for z==0
    change_positions_set = set()
    change_position2trigers = {}
    # criterion_mask_aggregated = np.zeros_like(df.index, dtype=bool)
    for criterion in criterions:
        if criterion == 'z':
            criterion_mask = df[criterion].values != 0
        elif criterion == 'tangential_velocity':
            criterion_mask = df[criterion].values >= 0
        
        
        change_positions = np.where(np.diff(np.concatenate(([0], criterion_mask.astype(int)))) != 0)[0]
        for change_position in change_positions:
            if change_position == 0:
                continue
            dct = change_position2trigers.get(change_position, {})
            dct.update({criterion: criterion_mask[change_position]})
            change_position2trigers[change_position] = dct

        dct = change_position2trigers.get(0, {})
        dct.update({criterion: criterion_mask[0]})
        change_position2trigers[0] = dct
        change_positions_set.update(change_positions)
    change_positions = sorted(change_positions_set)

    

    
    # Get segments with their start and end indices

    # segment_nums = change_positions.unique()
    # print(change_points)
    segments = []

    z_zero = True
    prev_z_zero = True
    next_z_zero = True
    tangential_positive = True

    for i, change_position in enumerate(change_positions):
        # print(group_num, df.index[group_num])
        start_idx = change_position
        # print(start_idx, group_num)
        # tangential_positive = False
        # z_zero = False
        # prev_z_zero = False
        # next_z_zero = False
        if 'tangential_velocity' in change_position2trigers[start_idx]:
            tangential_positive = change_position2trigers[start_idx]['tangential_velocity']
        if 'z' in change_position2trigers[start_idx]:
            z_zero = not change_position2trigers[start_idx]['z']
            if i > 0:
                if 'z' in change_position2trigers[change_positions[i - 1]]:
                    prev_z_zero = not change_position2trigers[change_positions[i - 1]]['z']
                else:
                    prev_z_zero = z_zero
            if i < len(change_positions) - 1:
                if 'z' in change_position2trigers[change_positions[i + 1]]:
                    next_z_zero = not change_position2trigers[change_positions[i + 1]]['z']
                else:
                    next_z_zero = z_zero
        
        if not z_zero and prev_z_zero:
            start_idx = max(start_idx - 1, 0)

        if i < len(change_positions) - 1:
            end_idx = change_positions[i + 1]
            if not (z_zero and not next_z_zero):
                end_idx = min(end_idx + 1, df.index[-1])
        else:
            end_idx = len(df)
        
        
        segment = df.iloc[start_idx:end_idx]
        # is_zero = df.iloc[end_idx]['z'] == 0
        # print(segment['z'].iloc[0], segment['z'].iloc[1], is_zero)

        segments.append({
            'start': start_idx,
            'end': end_idx,
            'z_zero': z_zero,
            'tangential_positive': tangential_positive,
            'data': segment
        })
    
    # Print segments info
    for i, seg in enumerate(segments):
        print(f"Segment {i}: {seg['z_zero']}, Index {seg['start']} to {seg['end']}")
    
    return segments

# %%


def _get_segment_color(segment):
    """Determine segment color based on its characteristics."""
    if segment['z_zero']:
        return 'blue' if segment['tangential_positive'] else 'red' # 'blue'
    return '#90D5FF' if segment['tangential_positive'] else '#FF7F7F' # '#90D5FF' # 

def _get_segment_style(segment):
    """Get plotting style for segment."""
    is_zero = segment['z_zero']
    return {
        'alpha': 1.0 if is_zero else 0.5,
        'marker': None if is_zero else 'o',
        'markersize': None if is_zero else 3
    }

def _configure_plot(ax, view_params):
    """Configure plot appearance."""
    ax.view_init(
        elev=view_params['elev'],
        azim=view_params['azim'],
        roll=view_params['roll']
    )
    
    ax.set_xlabel('X (pixels)', labelpad=60)
    ax.set_ylabel('Y (pixels)', labelpad=10, rotation=0)
    ax.set_zlabel('Z (pixels)', labelpad=10, rotation=0)
    ax.set_aspect('equal')
    ax.set_facecolor('white')

def draw_bracket_general(ax, x1, x2, y, height=100,
                         text=None, text_offset=6, lw=2, color='k', fontsize=34):
    """
    Draws a horizontal bracket on a matplotlib Axes object.

    Parameters:
    - ax: matplotlib Axes object where the bracket will be drawn.
    - x1: Starting x-coordinate of the bracket.
    - x2: Ending x-coordinate of the bracket.
    - y: The y-coordinate at which the bracket is drawn.
    - text: Optional text label to display above the bracket.
    - text_offset: Vertical offset for the text label.
    - lw: Line width of the bracket.
    - color: Color of the bracket.
    """
    # Bracket coordinates
    bx = [x1, x1, x2, x2]
    by = [y + height, y, y, y + height]

    # Draw bracket
    ax.plot(bx, by, color=color, lw=lw)

    # Add optional text above bracket
    if text:
        ax.text((x1 + x2) / 2, y + text_offset, text, 
                size=fontsize, ha='center', va='bottom', color=color)

def visualize_movement_segments(input_path, sampling_rate,
                              trial_number, direction_smoothing_kernel, 
                              time_range=(0, 10),
                              view_params={'elev': 40, 'azim': -122, 'roll': 0},
                              figsize=(12, 12), dpi=800):
    """
    Visualize movement segments with different colors based on movement characteristics.
    
    Args:
        input_path (str): Path to the input movement data CSV file
        sampling_rate (int): Target sampling rate for the data
        trial_number (int): Trial number to visualize
        direction_smoothing_kernel (int): Smoothing kernel size for direction calculation
        time_range (tuple): Time range (start, end) in seconds for visualization
        view_params (dict): Camera view parameters (elevation, azimuth, roll)
        figsize (tuple): Figure size in inches
        dpi (int): Figure DPI
    """
    # Generate output path based on input path
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    output_filename = f"tangential_velocity_{input_filename}"
    output_path = os.path.join(input_dir, output_filename)
    
    # Process data
    df_to_tangential_velocity(
        input_path, 
        output_path,
        target_sampling_rate=sampling_rate,
        direction_smoothing_kernel=direction_smoothing_kernel,
        direction_step=None
    )

    # Read and prepare data
    movement_df = pd.read_csv(output_path)
    
    # Add z column if it doesn't exist
    if 'z' not in movement_df.columns:
        movement_df['z'] = 0
    
    # Filter data for specific trial and time range
    trial_data = movement_df[
        (movement_df['trial_n'] == trial_number) & 
        (movement_df['time'] >= time_range[0]) &
        (movement_df['time'] <= time_range[1])
    ]
    
    if trial_data.empty:
        raise ValueError(f"No data found for trial {trial_number} in time range {time_range}")
    
    # Detect segments
    segments = detect_segments(trial_data)
    
    # Create visualization

    # set Times New Roman font
    plt.rcParams['font.family'] = 'Times New Roman'
    # increase font size
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 16


    fig0 = plt.figure(dpi=dpi, figsize=(12, 8))
    ax0 = fig0.add_subplot(111)

    # ax0.plot(trial_data['time'], trial_data['tangential_velocity'] * 370)


    fig = plt.figure(dpi=dpi, figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(212)

    # Plot segments
    for i, segment in enumerate(segments):
        df = segment['data']
        color = _get_segment_color(segment)
        style = _get_segment_style(segment)
        
        ax.plot(df['x'] - 900, -df['y'] + 100, df['z'],
               c=color,
               linewidth=3,
               alpha=style['alpha'],
               marker=style['marker'],
               markersize=style['markersize'])
        
        ax0.plot(df['time'], df['tangential_velocity'] * 370, c=color, linewidth=3)
    
        
        # write a mark and number at the beginning of the path
        # start_x, start_y, start_z = df['x'].iloc[0], -df['y'].iloc[0], df['z'].iloc[0]
        # ax.scatter(start_x, start_y, start_z,
        #           c=color,
        #           s=100,  # size of the marker
        #           marker='o')  # circle marker
        
        # ax.text(start_x, start_y, start_z,
        #         f' {i+1}',  # adding space before number for slight offset
        #         color=color,
        #         fontsize=10,
        #         fontweight='bold')
        
        # ax2.plot(df['time'], df['tangential_velocity'])
    
    # z ticks
    ax.set_zticks([0, 25, 50, 75])
    ax.set_zticklabels([0, 25, 50, 75])

    ax0.grid(True)
    ax0.set_xlabel('Time (s)', fontsize=26)
    ax0.set_ylabel('Tangential velocity (pixels/s)', fontsize=26)
    ax0.set_xlim(0, 1.85)

    draw_bracket_general(ax0, 0.21, 0.55, -2800, text="a")
    draw_bracket_general(ax0, 0.60, 0.79, -2800, text="p")
    draw_bracket_general(ax0, 0.95, 1.21, -2800, text="p")
    draw_bracket_general(ax0, 1.35, 1.54, -2800, text="r")
    draw_bracket_general(ax0, 1.62, 1.8, -2800, text="o")

    fig0.savefig('tangential_velocity.pdf', dpi=dpi)
    fig0.show()

    # ax.grid(False)
    # ax.set_axis_off()


    # Configure plot
    _configure_plot(ax, view_params)
    # plt.title(f'Direction Smoothing Kernel: {direction_smoothing_kernel}\n'
    #           f'Time Range: {time_range[0]}-{time_range[1]}s')

    
    return fig, ax



# ... rest of the helper functions remain the same ...

# Example usage:
if __name__ == "__main__":
    input_path = '/Users/NAME_REMOVED/data/human_movement/3d.3.writing/tablet_writing_data.csv'
    # '/Users/NAME_REMOVED/data/human_movement/3d.3.writing/tablet_writing_data.csv'
    #'/Users/NAME_REMOVED/data/human_movement/2d.2.whac-a-mole/whacamole_data.csv'
    #'/Users/NAME_REMOVED/data/human_movement/2d.1.Fitts_task/Fitts_data.csv'
    sampling_rate = 370
    
    for kernel_size in range(7, 8, 1):
        fig, ax = visualize_movement_segments(                                                                                                                                                         
            input_path=input_path,
            sampling_rate=sampling_rate,
            trial_number=293,
            direction_smoothing_kernel=kernel_size,
            time_range=(0, 10)  # visualize first 5 seconds
        )
        fig.savefig(f'movement_trial_293_kernel_{kernel_size}.pdf', dpi=300)
        plt.show()
# %%


for i in range(1, 6):
    tablet_writing_tangential_velocity_pandas_path = '/Users/NAME_REMOVED/data/human_movement/tablet_writing_tangential_velocity_data_high_sampling_rate3.csv'
    tablet_writing_pandas_path = '/Users/NAME_REMOVED/data/human_movement/3d.3.writing/tablet_writing_data.csv'
    df_to_tangential_velocity(tablet_writing_pandas_path, tablet_writing_tangential_velocity_pandas_path,
                          target_sampling_rate=370,
                          direction_smoothing_kernel=i,                                      
                          direction_step=None)

    # tablet_writing_tangential_velocity_pandas_path = '/Users/NAME_REMOVED/data/human_movement/tablet_writing_tangential_velocity_data_high_sampling_rate3.csv'

    tablet_writing_tangential_velocity_df = pd.read_csv(tablet_writing_tangential_velocity_pandas_path)
    tablet_writing_tangential_velocity_df.head()

    # tablet_writing_tangential_velocity_df['z'] = 0

    tablet_writing_df = tablet_writing_tangential_velocity_df
    # tablet_writing_df.head()

    # Use it

    # Access any segment's data like:
    # segments[0]['data']  # First segment

    
    tablet_writing_df_trial = tablet_writing_df[tablet_writing_df['trial_n'] == 293]

    tablet_writing_df_trial = tablet_writing_df_trial[tablet_writing_df_trial['time'] < 10]

    # tablet_writing_df_trial_hover = tablet_writing_df_trial[tablet_writing_df_trial['z'] > 0]
    # tablet_writing_df_trial_touch = tablet_writing_df_trial[tablet_writing_df_trial['z'] == 0]

    tablet_writing_df_trial_hover = tablet_writing_df_trial.copy()
    # tablet_writing_df_trial_hover[tablet_writing_df_trial_hover['z'] == 0][['x', 'y', 'z']] = np.nan
    tablet_writing_df_trial_touch = tablet_writing_df_trial.copy()
    # tablet_writing_df_trial_touch[tablet_writing_df_trial_touch['z'] > 0][['x', 'y', 'z']] = np.nan

    tablet_writing_df_trial_hover.head()
    segments = detect_segments(tablet_writing_df_trial)


    fig = plt.figure(dpi=800, figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for segment in segments:

        # if not segment['is_zero']:
        #     continue
        df = segment['data']
        if segment['z_zero']:
            if segment['tangential_positive']:
                color = 'blue'
            else:
                color = 'red'
        else:
            if segment['tangential_positive']:
                color = '#90D5FF'
            else:
                color = '#FF7F7F'

        alpha = 0.5 if not segment['z_zero'] else 1
        marker = 'o' if not segment['z_zero'] else None
        markersize = 3 if not segment['z_zero'] else None
        ax.plot(df['x'], -df['y'], df['z'], c=color, linewidth=3, 
                alpha=alpha,
                marker=marker,
                markersize=markersize)   

    # change view
    ax.view_init(elev=40, azim=-122, roll=0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # aspect ratio
    ax.set_aspect('equal')

    # remove background
    ax.set_facecolor('white')
    # ax.grid(False)
    # ax.set_axis_off()
    

    plt.title(i)
    # save to png
    fig.savefig('tablet_writing_trial_293.pdf', dpi=300)

    plt.show()

# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
tablet_writing_df_trial_positive = tablet_writing_df_trial.copy()
tablet_writing_df_trial_negative = tablet_writing_df_trial.copy()
tablet_writing_df_trial_positive[tablet_writing_df_trial_positive['tangential_velocity'] < 0] = np.nan
tablet_writing_df_trial_negative[tablet_writing_df_trial_negative['tangential_velocity'] >= 0] = np.nan
ax.plot(tablet_writing_df_trial_positive['x'], tablet_writing_df_trial_positive['y'], tablet_writing_df_trial_positive['z'], c='blue')
ax.plot(tablet_writing_df_trial_negative['x'], tablet_writing_df_trial_negative['y'], tablet_writing_df_trial_negative['z'], c='red')
ax.view_init(elev=40, azim=-122, roll=0)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# aspect ratio
ax.set_aspect('equal')

# remove background
ax.set_facecolor('white')
ax.grid(False)
ax.set_axis_off()


plt.show()

# %%
# same with plotly
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=tablet_writing_df_trial['x'], y=tablet_writing_df_trial['y'], z=tablet_writing_df_trial['z'], mode='markers', marker=dict(color=tablet_writing_df_trial['tangential_velocity'] >= 0)))
fig.show()

# %%
plt.plot(tablet_writing_df_trial['time'], tablet_writing_df_trial['tangential_velocity'])
plt.plot(tablet_writing_df_trial['time'], tablet_writing_df_trial['tangential_velocity'].abs())
plt.grid(True)
plt.show()

# %%

plt.subplot(4, 1, 1)
plt.plot(tablet_writing_df_trial['time'], tablet_writing_df_trial['x'])
plt.subplot(4, 1, 2)
plt.plot(tablet_writing_df_trial['time'], tablet_writing_df_trial['y'])
plt.subplot(4, 1, 3)
plt.plot(tablet_writing_df_trial['time'], tablet_writing_df_trial['z'])
# plt.subplot(4, 1, 4)
#plt.plot(tablet_writing_df_trial['time']

# %%
tablet_writing_df_tangential_velocity = pd.read_csv('/Users/NAME_REMOVED/data/human_movement/tablet_writing_tangential_velocity_data.csv')
tablet_writing_df_tangential_velocity_trial = tablet_writing_df_tangential_velocity[tablet_writing_df_tangential_velocity['trial_n'] == 4657]


# %%
plt.figure(dpi=300)
plt.subplot(4, 1, 1)
plt.grid(True)
plt.plot(tablet_writing_df_tangential_velocity_trial['time'], tablet_writing_df_tangential_velocity_trial['tangential_velocity'])
plt.subplot(4, 1, 2)
plt.grid(True)
plt.plot(tablet_writing_df_tangential_velocity_trial['time'], tablet_writing_df_tangential_velocity_trial['x'].diff())
plt.subplot(4, 1, 3)
plt.grid(True)
plt.plot(tablet_writing_df_tangential_velocity_trial['time'], tablet_writing_df_tangential_velocity_trial['y'].diff())
plt.subplot(4, 1, 4)
plt.grid(True)
plt.plot(tablet_writing_df_tangential_velocity_trial['time'], tablet_writing_df_tangential_velocity_trial['z'].diff())

plt.tight_layout()
plt.show()

# %%
pointing_pandas_path = '/Users/NAME_REMOVED/data/human_movement/3d.2.pointing/pointing_data.csv'
pointing_df = pd.read_csv(pointing_pandas_path)

# %%
pointing_df_trial = pointing_df[pointing_df['trial_n'] == 9]
pointing_df_trial.head()

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pointing_df_trial['x'], pointing_df_trial['y'], pointing_df_trial['z'])
plt.show()

# %%
pointing_df_participant = pointing_df[pointing_df['participant'] == 'D']
pointing_df_participant.head()

# %%
fig = plt.figure(dpi=800, figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

unique_trials = pointing_df_participant['trial_n'].unique()

for trial in unique_trials:
    pointing_df_trial = pointing_df_participant[pointing_df_participant['trial_n'] == trial]
    if trial == 1136:
        color = 'blue'
        alpha = 1
        linewidth = 5
    else:
        color = '#606060'  # dark gray
        alpha = 0.2
        linewidth = 2
    ax.plot(pointing_df_trial['x'], pointing_df_trial['y'], pointing_df_trial['z'], color=color, alpha=alpha, linewidth=linewidth)

# change view
ax.view_init(elev=10, azim=60, roll=0)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# aspect ratio
ax.set_aspect('equal')

# remove background
ax.set_facecolor('white')
ax.grid(False)
ax.set_axis_off()

# save to png
fig.savefig('pointing_trial_1136.png', dpi=300)

plt.show()


# %%
# plot with plotly
import plotly.graph_objects as go

fig = go.Figure()
for trial in unique_trials:
    if trial == 1136:
        color = 'blue'
        alpha = 1
        linewidth = 20
    else:
        color = 'black'
        alpha = 0.15
        linewidth = 15
    pointing_df_trial = pointing_df_participant[pointing_df_participant['trial_n'] == trial]
    fig.add_trace(go.Scatter3d(x=pointing_df_trial['x'] - pointing_df_participant['x'].min(),
                                   y=pointing_df_trial['y'] - pointing_df_participant['y'].min(),
                                   z=pointing_df_trial['z'] - pointing_df_participant['z'].min(),
                                   mode='lines',
                                   opacity=alpha,
                                   line=dict(width=linewidth, color=color),
                                   showlegend=False,
                                   name=f'Trial {trial}'
                                   ))
                                   

# remove background
fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))

# save to html
fig.write_html('pointing_trial_1136.html')
