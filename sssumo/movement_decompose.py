""" Adapted from https://github.com/JasonFriedman/submovements
A module to decompose a 2d movement to multiple submovements
Contain the following functions:
    load_data            - read a folder full of csv files, 
                           collect movements position, velocities & recorded time
    plot_position        - plot movement position in time
    plot_velocity        - plot movement velocity in time
    decompose            - estimate submovements parameters from 1D, 2D, or 3D movement
    decompose_2D         - estimate submovements parameters from 2D movement
    plot_submovements_2D - plot the expected velocities from submovement group

The python code is based on the Matlab code, and was mostly written during a Hackathon:
https://github.com/Liordemarcas/submovements

by
Omer Ophir:             https://github.com/omerophir
Omri FK:                https://github.com/OmriFK
Shay Eylon:             https://github.com/ShayEylon
Lior de Marcas (LdM):   https://github.com/Liordemarcas

Updated by Jason Friedman https://github.com/JasonFriedman
"""

ERROR_FUNCTION = 'absolute' # 'absolute_relative', 'absolute', 'squared'

import os
import re
import time
import math 

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter
from scipy.optimize import minimize
import scipy.optimize

import torch

def generate_and_adjust_params(bounds):
    """
    Generates random initial parameters and adjusts them based on the provided bounds.

    Parameters:
    bounds (list of lists): A list where each sublist contains the lower and upper bounds for each parameter.

    Returns:
    list: The adjusted initial parameters.
    """
    # Determine the number of parameter types and total number of parameters
    total_params = len(bounds)
    # total_params = sum(len(param_bounds) for param_bounds in bounds)
    
    # Generate random initial parameters between 0 and 1
    initial_params = np.random.rand(total_params)
    
    # Adjust the parameters based on the bounds
    for param_index, param_bounds in enumerate(bounds):
        lower_bound, upper_bound = param_bounds
        # Scale and shift the parameter to fit within the specified bounds
        initial_params[param_index] *= (upper_bound - lower_bound)
        initial_params[param_index] += lower_bound
    
    return initial_params


def fill_zeros_with_nearest_nonzero(vector):

    if np.sum(np.abs(vector)) == 0:
        return vector

    non_zero_indices = np.where(vector != 0)[0]
    filled_vector = vector.copy()
    
    for i, value in enumerate(vector):
        if value == 0:
            nearest_idx = non_zero_indices[np.abs(non_zero_indices - i).argmin()]
            filled_vector[i] = vector[nearest_idx]
    
    return filled_vector


def decompose_pytorch_style(x, 
                            mask_true=None,
                             window_size: float = 120,
                             step_size: float = 60,
                             max_submovements_per_window: int = 24, # maybe overkill
                             displacement_rng: tuple = (-30., 30.),
                             duration_rng: tuple = (5, 60),
                             n_iter: int = 10,
                             scale_error: bool = True,
                             optimizer: str = 'L-BFGS-B',
                             options: dict | None = None,
                             limit_clutters: bool = False,
                             error_threshold: float = 0.05,
                             exact_number_of_submovements: bool = False,
                             patience: int = 2,
                             debug_level: int = 0, # 0 - no debug, 1 - only the final message, 2 -  within current function, 3 - full debug
                             ):
    

    
    x = x.squeeze(1) # remove the channel dimension

    y_pred = []
    for i, x_el in enumerate(x):
        if debug_level >= 1:
            print(f'Processing batch element {i+1}/{len(x)}')

        # if mask_true is not None:
        #     mask_true_el = mask_true[i].numpy()
        # else:
        #     mask_true_el = None
        y_el_pred = decompose_return_mask(x_el.numpy().copy(), 
                                         t=None,
                                         window_size=window_size,
                                         step_size=step_size,
                                         max_submovements_per_window=max_submovements_per_window,
                                         displacement_rng=displacement_rng,
                                         duration_rng=duration_rng,
                                         n_iter=n_iter,
                                         scale_error=scale_error,
                                         optimizer=optimizer,
                                         options=options,
                                         limit_clutters=limit_clutters,
                                         error_threshold=error_threshold,
                                         exact_number_of_submovements=exact_number_of_submovements,
                                         patience=patience,
                                         debug_level=debug_level)
        y_pred.append(y_el_pred)

    y_pred = np.stack(y_pred, axis=0)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    return y_pred


def decompose_return_mask(vel: np.ndarray, 
                             t: np.ndarray | None = None,
                             window_size: float = 120,
                             step_size: float = 60,
                             max_submovements_per_window: int = 24, # maybe overkill
                             displacement_rng: tuple = (-30., 30.),
                             duration_rng: tuple = (5, 60),
                             n_iter: int = 10,
                             scale_error: bool = True,
                             optimizer: str = 'L-BFGS-B',
                             options: dict | None = None,
                             limit_clutters: bool = False,
                             error_threshold: float = 0.05,
                             exact_number_of_submovements: bool = False,
                             patience: int = 2,
                             debug_level: int = 0, # 0 - no debug, 1 - only the final message, 2 -  within current function, 3 - full debug
                             ):
    
    params, _ = decompose_sliding_window(vel, 
                                         t=t,
                                         window_size=window_size,
                                         step_size=step_size,
                                         max_submovements_per_window=max_submovements_per_window,
                                         displacement_rng=displacement_rng,
                                         duration_rng=duration_rng,
                                         n_iter=n_iter,
                                         scale_error=scale_error,
                                         optimizer=optimizer,
                                         options=options,
                                         limit_clutters=limit_clutters,
                                         error_threshold=error_threshold,
                                         exact_number_of_submovements=exact_number_of_submovements,
                                         patience=patience,
                                         debug_level=debug_level)
    

    mask_array = np.zeros_like(vel)
    amplitude_array = np.zeros_like(vel)
    duration_array = np.zeros_like(vel)

    # previous_onset = -1

    for single_submovement_params in params:


        onset = single_submovement_params[0]
        onset = round(onset)
        duration = single_submovement_params[1]
        amplitude = single_submovement_params[2]

        if onset >= len(mask_array):
            continue

        mask_array[onset] = 1
        amplitude_array[onset] = amplitude
        duration_array[onset] = duration


    amplitude_array = fill_zeros_with_nearest_nonzero(amplitude_array)
    duration_array = fill_zeros_with_nearest_nonzero(duration_array)

    mask_stacked = np.stack([mask_array, amplitude_array, duration_array], axis=0)
    
    return mask_stacked


def decompose_sliding_window(vel: np.ndarray, 
                             t: np.ndarray | None = None,
                             window_size: float = 120,
                             step_size: float = 60,
                             max_submovements_per_window: int = 24, # maybe overkill
                             displacement_rng: tuple = (-30., 30.),
                             duration_rng: tuple = (5, 60),
                             n_iter: int = 10,
                             scale_error: bool = True,
                             optimizer: str = 'L-BFGS-B',
                             options: dict | None = None,
                             limit_clutters: bool = False,
                             error_threshold: float = 0.05,
                             exact_number_of_submovements: bool = False,
                             patience: int = 2,
                             debug_level: int = 0, # 0 - no debug, 1 - only the final message, 2 -  within current function, 3 - full debug
                             ) -> tuple[float, np.ndarray]:
    
    if t is None:
        t = np.arange(len(vel))

    params = []

    if scale_error:
        abs_mean_vel = np.mean(np.abs(vel))
        if abs_mean_vel != 0:
            vel /= abs_mean_vel

    vel_subtracted = vel.copy()

    tm_optimization_start = time.time()

    for segment_start in np.arange(t[0], t[-1] - step_size, step_size):

        tm_start = time.time()
        segment_end = segment_start + window_size
        if segment_end > t[-1]:
            optimized_end = t[-1]
        else:
            optimized_end = segment_start + step_size
        segment_t = t[(t >= segment_start) & (t <= optimized_end)]
        segment_duration = segment_t[-1] - segment_t[0]
        segment_vel = vel_subtracted[(t >= segment_start) & (t <= optimized_end)]

        current_max_submovements_number = math.ceil(max_submovements_per_window * segment_duration / window_size)

        current_error, current_params = decompose_1D(vel=segment_vel, 
                                               t=segment_t, 
                                               max_submovements_number=current_max_submovements_number, 
                                               displacement_rng=displacement_rng, 
                                               duration_rng=duration_rng, 
                                               n_iter=n_iter, 
                                               scale_error=False, 
                                               optimizer=optimizer, 
                                               options=options, 
                                               limit_clutters=limit_clutters,
                                               error_threshold=error_threshold,
                                               exact_number_of_submovements=exact_number_of_submovements,
                                               patience=patience,
                                               debug=debug_level >= 3
                                            )
        
        tm_end = time.time()
        
        if len(current_params) > 0:
            params_to_keep = current_params[current_params[:, 0] <= optimized_end]
        else:
            params_to_keep = []
        params.extend(params_to_keep)

        vel_segment_reconstructed = params_to_velocity(params_to_keep, t)
        vel_subtracted -= vel_segment_reconstructed

        if debug_level >= 2:
            print(f'Window {segment_start} - {segment_end} with error {current_error} and {len(params_to_keep)} submovements in {tm_end - tm_start} seconds')

            plt.plot(t, vel, label='original')
            plt.plot(t, vel_subtracted, label='subtracted')
            plt.plot(t, vel_segment_reconstructed, label='reconstructed')
            plt.legend()
            plt.show()

    params = np.array(params)

    if scale_error and params.size > 0:
        params[:, 2] *= abs_mean_vel
        vel *= abs_mean_vel
        vel_subtracted *= abs_mean_vel

    t_step = (t[-1] - t[0]) / (t.size - 1)

    if t_step == 1 and params.size > 0:
        params[:, 0] += 0.5
        params[:, 0] = np.round(params[:, 0])
        params[:, 0] -= 0.5

    previous_onset = -1
    for i, param_line in enumerate(params):
        onset = param_line[0]

        if onset <= previous_onset:
            onset = previous_onset + 1

        previous_onset = onset

        params[i, 0] = onset

    params = sorted(params, key=lambda x: x[0])
    params = np.array(params)
    error = _calculate_error_MJ1D(params, t, vel)

    if t_step == 1 and params.size > 0:
        params[:, 0] += 0.5
    


    if debug_level >= 1:
        print(f'Total time for optimization: {time.time() - tm_optimization_start} seconds, error: {error}, number of submovements: {len(params)}')

        plt.plot(t, vel, label='original')
        plt.plot(t, params_to_velocity(params, t), label='reconstructed')
        plt.plot(t, vel_subtracted, label='residual')
        plt.legend()
        plt.show()
        print(params)

    
    return params, error



def remove_cluttered_submovements(params: np.ndarray, min_distance=3):
    if params.size == 0:
        return params
    
    distance_delta = np.zeros(params.shape[0])
    distance_delta[0] = min_distance
    distance_delta[1:] = params[1:, 0] - params[:-1, 0]

    params = params[distance_delta >= min_distance]

    return params
    

def decompose_1D(vel: np.ndarray, 
                t: np.ndarray | None = None,
                max_submovements_number: int = 4,
                submovements_rate: float | None = None,
                displacement_rng: tuple = (-30., 30.),
                duration_rng: tuple = (5, 60),
                n_iter: int = 10,
                scale_error: bool = False,
                optimizer: str = 'L-BFGS-B',
                options: dict | None = None,
                limit_clutters: bool = False, # in most cases, it will make it worse, cause of lowering probability of having submovs early in the vel
                error_threshold: float = 0.1,
                exact_number_of_submovements: bool = False,
                patience: int = 3,
                debug: bool = True,
                ) -> tuple[float, np.ndarray]:
    """
    decompose_1D - decompose one dimensional movement into submovements using the velocity profiles
    
    best_error, final_params = decompose_1D(time, vel, n_sub_movement, rng)
    
    Parameters:
    -----------
    time : np.ndarray
        A 1D array with the corresponding time (in seconds)
    
    vel : np.ndarray
        A 1D array with the velocity
    
    n_sub_movement : int, optional
        The number of submovements to look for (default = 4)
    
    rng : tuple, optional
        The valid range for the amplitude values (default = (-5, 5))
    
    duration_rng : tuple, optional
        The valid range for the duration values (default = (0.083, 1.0))
    
    n_iter : int, optional
        The number of iterations to run (default = 20)
    
    Returns:
    --------
    best_error : float
        The best (lowest) value of the error function
    
    final_params : np.ndarray
        The function parameters corresponding to the best values
        Each row contains [t0, D, A] where:
        - t0 is the start time
        - D is the duration
        - A is the amplitude
    """

    if t is None:
        t = np.arange(len(vel))

    NUM_PARAMS_PER_SUBMOVEMENT = 3

    BOUND = [(min(t), max(t)), 
             (duration_rng[0], duration_rng[1]), 
             (displacement_rng[0], displacement_rng[1])]
    
    if submovements_rate is not None:
        max_submovements_number = int((t[-1] - t[0]) * submovements_rate)

    if exact_number_of_submovements:
        n_submovements = max_submovements_number
    else:
        n_submovements = 1

    # Input validation
    if t.ndim > 1:
        raise ValueError('t must be a 1D array')
    
    if vel.ndim > 1:
        if vel.shape[1] != 1:
            raise ValueError('vel must be a 1D array or Nx1 array')
        vel = vel.flatten()
    
    if vel.size != t.size:
        raise ValueError('vel must match t length')
    
    original_options = {'gtol': 1e-5, 'disp': False, 'maxiter': 100} 
    if options is not None:
        original_options.update(options)
        options = original_options
        

    
    lower_duration = duration_rng[0]
    upper_duration = duration_rng[1]
    
    # Set lower and upper bounds for the parameters [t0, D, A]
    lower_bounds = np.array([t[0], lower_duration, displacement_rng[0]])  # [start_time, duration, amplitude]
    #upper_bounds = np.array([max(time[-1]-lower_duration, 0.1), upper_duration, rng[1]])
    upper_bounds = np.array([t[-2], upper_duration, displacement_rng[1]])
    
    if np.any(lower_bounds > upper_bounds):
        raise ValueError('Lower bounds exceed upper bounds - infeasible')
    

    best_error = np.inf
    best_params = None

    if scale_error:
        # vel /= len(vel)
        abs_mean_vel = np.mean(np.abs(vel))
        vel /= abs_mean_vel
        # hack to scale the error function, to make it comparable for different velocities 

    error_fun = lambda params: _calculate_error_MJ1D(params, t, vel, timedelta=0.005)
    calculate_jacobian = lambda params: _calculate_Jacobian_MJ1D(params, t, vel, timedelta=0.005)



    # min_delay = lower_duration
    # total_duration = t[-1] - t[0]
    # if min_delay >= upper_bounds[0] / n_sub_movement:
    #     min_delay = upper_bounds[0] / n_sub_movement

    best_params = np.array([])
    best_error = error_fun(best_params)
    best_error_per_number_of_submovements = [best_error] + [np.inf] * max_submovements_number
    best_number_of_submovements = 0

    bound = np.stack([lower_bounds, upper_bounds], axis=1) 
    bounds = np.tile(bound, (n_submovements, 1))

    if debug:
        print(f'The optimization is started in the range of {n_submovements}:{max_submovements_number} submovements')
        print(f'The bound is {bound}')

    while best_error > error_threshold and n_submovements <= max_submovements_number:

        for i in range(n_iter):
            init_parm = generate_and_adjust_params(bounds)

            optimization_result = scipy.optimize.minimize(
                error_fun,
                init_parm,
                method=optimizer,
                jac=calculate_jacobian,
                bounds=bounds,
                options=options
            )

            current_params = optimization_result.x.reshape(n_submovements, NUM_PARAMS_PER_SUBMOVEMENT)
            current_error = optimization_result.fun
            
            # Update best parameters if we found a better solution

            if current_error < best_error:
                best_error = current_error
                best_params = current_params
                best_number_of_submovements = n_submovements

                if debug:
                    print(f'Iteration {i} with {n_submovements} submovements, error: {current_error}')
                    print(optimization_result)


            if current_error < best_error_per_number_of_submovements[n_submovements]:
                best_error_per_number_of_submovements[n_submovements] = current_error

        
        # first_submovement2check = 0
        if best_number_of_submovements <= n_submovements - patience:
            if debug:
                print(f'No improvement for {patience} submovements, stopping the optimization.')
            break
        n_submovements += 1
        bounds = np.tile(bound, (n_submovements, 1))


    best_params = np.array(sorted(best_params, key=lambda x: x[0]))



    if debug:
        print(f'Optimization finished with {n_submovements} submovements, error: {best_error}')
        print(f'Best parameters: {best_params}')


    if scale_error:
        best_params[:, 2] *= abs_mean_vel
        
    return best_error, best_params

def _calculate_error_MJ1D(parameters: np.ndarray, t: np.ndarray, 
                         vel: np.ndarray, timedelta: float = 0.005) -> float:
    """
    Calculate the error between the observed and predicted velocities for 1D signals
    
    Parameters:
    -----------
    parameters : np.ndarray
        Array of parameters for submovements, each row: [t0, D, A]
    time : np.ndarray
        Time points
    vel : np.ndarray
        Observed velocity
    timedelta : float, optional
        Time resolution for calculations
    
    Returns:
    --------
    error : float
        Error between observed and predicted velocities
    """
    # Reshape parameters if flattened
    num_params = 3  # [t0, D, A]
    if parameters.ndim == 1:
        n_submovements = parameters.size // num_params
        parameters = parameters.reshape(n_submovements, num_params)
    elif len(parameters) == 0:
        parameters = np.array([])
        n_submovements = 0
    
    # Initialize sum of predicted velocities
    sum_predicted = np.zeros_like(vel)
    
    # Add contribution from each submovement
    for i in range(parameters.shape[0]):
        t0, D, A = parameters[i, :]
        sub_vel = _minimum_jerk_velocity_1D(t0, D, A, t)
        sum_predicted += sub_vel
    
    # Calculate squared error
    # if ERROR_FUNCTION == 'absolute_relative':
    #     error = np.sum(np.abs(vel - sum_predicted) / vel)
    if ERROR_FUNCTION == 'absolute':
        error = np.sum(np.abs(vel - sum_predicted))
    elif ERROR_FUNCTION == 'squared':
        error = np.sum((vel - sum_predicted) ** 2)

    error /= vel.size
    
    return error

def _calculate_Jacobian_MJ1D(parameters: np.ndarray, t: np.ndarray, 
                            vel: np.ndarray, timedelta: float = 0.005) -> np.ndarray:
    """
    Calculate the Jacobian for the error function in 1D signals
    
    Parameters:
    -----------
    parameters : np.ndarray
        Array of parameters for submovements, each row: [t0, D, A]
    time : np.ndarray
        Time points
    vel : np.ndarray
        Observed velocity
    timedelta : float, optional
        Time resolution for calculations
    
    Returns:
    --------
    jacobian : np.ndarray
        Jacobian of the error function
    """
    # Reshape parameters if flattened
    num_params = 3  # [t0, D, A]
    if parameters.ndim == 1:
        n_submovements = parameters.size // num_params
        parameters = parameters.reshape(n_submovements, num_params)
    
    # Initialize sum of predicted velocities and Jacobians
    sum_predicted = np.zeros_like(vel)
    sumJ = np.zeros((parameters.shape[0] * num_params, vel.size))
    
    # Add contribution from each submovement
    for i in range(parameters.shape[0]):
        t0, D, A = parameters[i, :]
        
        # Calculate velocity and Jacobian for this submovement
        sub_vel = _minimum_jerk_velocity_1D(t0, D, A, t)
        sub_J = _minimum_jerk_Jacobian_1D(t0, D, A, t)
        
        # Add to total velocity
        sum_predicted += sub_vel
        
        # Store Jacobian components
        start_idx = i * num_params
        sumJ[start_idx:start_idx + num_params, :] = sub_J.T
    
    # Difference between observed and predicted
    diff = vel - sum_predicted
    
    # Calculate Jacobian of the error function
    sum_traj_sq = vel.size
    jacobian = np.zeros(parameters.shape[0] * num_params)
    
    for i in range(jacobian.size):
        if ERROR_FUNCTION == 'absolute':
            jacobian[i] = -1/sum_traj_sq * np.sum(np.sign(diff) * sumJ[i, :])  # For absolute error
        elif ERROR_FUNCTION == 'squared':
            jacobian[i] = -2/sum_traj_sq * np.sum(diff * sumJ[i, :]) # For squared error
    
    return jacobian

def _calculate_Hessian_MJ1D(parameters: np.ndarray, t: np.ndarray, 
                           vel: np.ndarray, timedelta: float = 0.005) -> np.ndarray:
    """
    Calculate the Hessian for the error function in 1D signals
    
    Parameters:
    -----------
    parameters : np.ndarray
        Array of parameters for submovements, each row: [t0, D, A]
    time : np.ndarray
        Time points
    vel : np.ndarray
        Observed velocity
    timedelta : float, optional
        Time resolution for calculations
    
    Returns:
    --------
    hessian : np.ndarray
        Hessian of the error function
    """
    # Reshape parameters if flattened
    num_params = 3  # [t0, D, A]
    if parameters.ndim == 1:
        n_submovements = parameters.size // num_params
        parameters = parameters.reshape(n_submovements, num_params)
    
    # Initialize sum of predicted velocities, Jacobians, and Hessians
    sum_predicted = np.zeros_like(vel)
    sumJ = np.zeros((parameters.shape[0] * num_params, vel.size))
    sumH = np.zeros((parameters.shape[0] * num_params, parameters.shape[0] * num_params, vel.size))
    
    # Add contribution from each submovement
    for i in range(parameters.shape[0]):
        t0, D, A = parameters[i, :]
        
        # Calculate velocity, Jacobian, and Hessian for this submovement
        sub_vel = _minimum_jerk_velocity_1D(t0, D, A, t)
        sub_J = _minimum_jerk_Jacobian_1D(t0, D, A, t)
        sub_H = _minimum_jerk_Hessian_1D(t0, D, A, t)
        
        # Add to total velocity
        sum_predicted += sub_vel
        
        # Store Jacobian and Hessian components
        start_idx = i * num_params
        sumJ[start_idx:start_idx + num_params, :] = sub_J.T
        
        for j in range(num_params):
            for k in range(num_params):
                sumH[start_idx + j, start_idx + k, :] = sub_H[:, j, k]
    
    # Difference between observed and predicted
    diff = vel - sum_predicted
    
    # Calculate Hessian of the error function
    sum_traj_sq = vel.size
    hessian = np.zeros((parameters.shape[0] * num_params, parameters.shape[0] * num_params))
    
    for i in range(hessian.shape[0]):
        for j in range(hessian.shape[1]):
            hessian[i, j] = 2/sum_traj_sq * np.sum(
                sumJ[i, :] * sumJ[j, :] + diff * sumH[i, j, :]
            )
    
    return hessian

def _minimum_jerk_velocity_1D(t0: float, D: float,
                             A: float,
                             t: np.ndarray) -> np.ndarray:
    """
    minimumJerkVelocity1D - evaluate a minimum jerk velocity curve for a 1D signal
    
    See Flash and Hogan (1985) for details on the minimum jerk equation
    
    Parameters:
    -----------
    t0 : float
        Movement start time
    D : float
        Movement duration
    A : float
        Displacement resulting from the movement
    t : np.ndarray
        Time points at which to evaluate the function
        
    Returns:
    --------
    vel : np.ndarray
        Velocity profile of the minimum jerk trajectory
    """

    normalized_time = (t - t0) / D
    logical_movement = (normalized_time >= 0) & (normalized_time <= 1)
    
    # normalize displacement to movement duration
    norm_disp = A / D
    # norm_disp = A # Here I don't normalize displacement
    
    # create velocity array (zero outside the movement timeframe)
    vel = np.zeros(t.size)
    
    # calculate velocity within the movement timeframe
    # polynomial function from Flash and Hogan (1985)
    vel[logical_movement] = norm_disp * (-60 * normalized_time[logical_movement]**3 
                                         + 30 * normalized_time[logical_movement]**4 
                                         + 30 * normalized_time[logical_movement]**2)
    
    return vel

def _minimum_jerk_Jacobian_1D(t0: float, D: float,
                             A: float,
                             t: np.ndarray) -> np.ndarray:
    """
    minimumJerkJacobian1D - evaluate the Jacobian (partial derivative) of a 
    minimum jerk velocity curve for a 1D signal
    
    Parameters:
    -----------
    t0 : float
        Movement start time
    D : float
        Movement duration
    A : float
        Displacement resulting from the movement
    t : np.ndarray
        Time points at which to evaluate the function
        
    Returns:
    --------
    J : np.ndarray
        Jacobian matrix of the velocity profile with respect to parameters [t0, D, A]
    """
    # normalize time to t0 and movement duration


    # t_step = (t[-1] - t[0]) / (t.size - 1)
    # if t_step == 1:
    #      t0 = round(t0 + 0.5)
    #      t0 -= 0.5
    # normalized_time = (t - t0 + t_step / 2) / (D + t_step)
    # normalized_time = np.clip(normalized_time, 0, 1)
    # normalized_time = np.nan_to_num(normalized_time, 0)

    normalized_time = (t - t0) / D

    logical_movement = (normalized_time >= 0) & (normalized_time <= 1)
    
    # Create Jacobian matrix (parameters: t0, D, A)
    J = np.zeros((t.size, 3))
    
    # Compute partial derivatives within movement timeframe
    # With respect to t0
    J[logical_movement, 0] = -A * ((1.0 / D**3 * (t[logical_movement]*2.0 - t0*2.0) * 30) 
                                   - (1.0 / D**4 * (t[logical_movement] - t0)**2 * 180) 
                                   + (1.0 / D**5 * (t[logical_movement] - t0)**3 * 120))
    
    # With respect to D
    J[logical_movement, 1] = -A * ((1.0 / D**4 * (t[logical_movement] - t0)**2 * 90) 
                                   - (1.0 / D**5 * (t[logical_movement] - t0)**3 * 240) 
                                   + (1.0 / D**6 * (t[logical_movement] - t0)**4 * 150))
    
    # With respect to A
    J[logical_movement, 2] = ((1.0 / D**3 * (t[logical_movement] - t0)**2 * 30) 
                             - (1.0 / D**4 * (t[logical_movement] - t0)**3 * 60) 
                             + (1.0 / D**5 * (t[logical_movement] - t0)**4 * 30))
    
    return J

def _minimum_jerk_Hessian_1D(t0: float, D: float,
                           A: float,
                           t: np.ndarray) -> np.ndarray:
    """
    minimumJerkHessian1D - evaluate the Hessian (second-order partial derivatives) 
    of a minimum jerk velocity curve for a 1D signal
    
    Parameters:
    -----------
    t0 : float
        Movement start time
    D : float
        Movement duration
    A : float
        Displacement resulting from the movement
    t : np.ndarray
        Time points at which to evaluate the function
        
    Returns:
    --------
    H : np.ndarray
        Hessian tensor of the velocity profile with respect to parameters [t0, D, A]
    """
    # normalize time to t0 and movement duration
    normalized_time = (t - t0) / D
    logical_movement = (normalized_time >= 0) & (normalized_time <= 1)
    
    # Create Hessian tensor (3x3 matrix for each time point)
    H = np.zeros((t.size, 3, 3))
    
    # Compute second-order partial derivatives within movement timeframe
    # With respect to t0,t0
    H[logical_movement, 0, 0] = A * ((1.0 / D**4 * (t[logical_movement]*2.0 - t0*2.0) * -180) 
                                    + (1.0 / D**5 * (t[logical_movement] - t0)**2 * 360) 
                                    + (1.0 / D**3 * 60))
    
    # With respect to t0,D
    H[logical_movement, 0, 1] = A * ((1.0 / D**4 * (t[logical_movement]*2.0 - t0*2.0) * 90) 
                                    - (1.0 / D**5 * (t[logical_movement] - t0)**2 * 720) 
                                    + (1.0 / D**6 * (t[logical_movement] - t0)**3 * 600))
    
    # With respect to t0,A
    H[logical_movement, 0, 2] = ((1.0 / D**3 * (t[logical_movement]*2.0 - t0*2.0) * -30) 
                               + (1.0 / D**4 * (t[logical_movement] - t0)**2 * 180) 
                               - (1.0 / D**5 * (t[logical_movement] - t0)**3 * 120))
    
    # With respect to D,t0 (symmetric)
    H[logical_movement, 1, 0] = H[logical_movement, 0, 1]
    
    # With respect to D,D
    H[logical_movement, 1, 1] = A * ((1.0 / D**5 * (t[logical_movement] - t0)**2 * 360) 
                                    - (1.0 / D**6 * (t[logical_movement] - t0)**3 * 1200) 
                                    + (1.0 / D**7 * (t[logical_movement] - t0)**4 * 900))
    
    # With respect to D,A
    H[logical_movement, 1, 2] = ((1.0 / D**4 * (t[logical_movement] - t0)**2 * -90) 
                               + (1.0 / D**5 * (t[logical_movement] - t0)**3 * 240) 
                               - (1.0 / D**6 * (t[logical_movement] - t0)**4 * 150))
    
    # With respect to A,t0 (symmetric)
    H[logical_movement, 2, 0] = H[logical_movement, 0, 2]
    
    # With respect to A,D (symmetric)
    H[logical_movement, 2, 1] = H[logical_movement, 1, 2]
    
    # With respect to A,A (all zeros, as the function is linear in A)
    
    return H 

def params_to_velocity(parameters: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    params_to_velocity - convert parameters to velocity
    """
    

    
    if len(parameters) == 0:
        return np.zeros(t.size)
    elif parameters.ndim == 1:
        parameters = parameters.reshape(1, -1)
    

    parameters = np.array(parameters)
    n_submovements = parameters.shape[0]
    t0 = parameters[:, 0]  # start times
    D = parameters[:, 1]   # durations
    A = parameters[:, 2]   # amplitudes
    
    # Make sure parameters are ordered by movement start time
    order = np.argsort(t0)

    vel = np.zeros((n_submovements, t.size))


    # Using minimum jerk, find velocity curve for each submovement
    for i_sub in range(n_submovements):
        vel[i_sub, :] = _minimum_jerk_velocity_1D(t0[i_sub], D[i_sub], A[i_sub], t)
    
    # Get total velocity expected from submovements
    sum_vel = np.sum(vel, axis=0)
    return sum_vel
    

def plot_submovements_1D(parameters, t: np.ndarray = None, plot_type: int = 1) -> tuple:
    """
    plot_submovements_1D - plot 1D submovements after decomposition

    Parameters:
    -----------
    parameters : np.ndarray
        Each row contains parameters for one submovement [t0, D, A]
        where:
        - t0: start time
        - D: duration
        - A: amplitude
    
    t : np.ndarray, optional
        Time points at which to evaluate the submovements.
        If None, will create a time vector based on the submovements.
    
    plot_type : int, optional
        Type of plot to generate (default = 1)
        1 = time vs submovement velocity + sum velocity (default)
        2 = time vs submovement velocity only
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes object containing the plot
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    vel_lines : list
        List of line objects for individual submovements
    vel_sum_line : list or None
        Line object for the summed velocity (None if plot_type=2)
    """
    if parameters.ndim == 1:
        # Handle the case where a single submovement is passed as a 1D array
        parameters = parameters.reshape(1, -1)
    
    if parameters.shape[1] != 3:
        raise ValueError('Parameters must have 3 columns [t0, D, A]')
    
    # Extract parameters
    n_submovements = parameters.shape[0]
    t0 = parameters[:, 0]  # start times
    D = parameters[:, 1]   # durations
    A = parameters[:, 2]   # amplitudes
    
    # Make sure parameters are ordered by movement start time
    order = np.argsort(t0)
    t0 = t0[order]
    D = D[order]
    A = A[order]
    
    # If no time was given, plot from start of first movement to end of last movement
    if t is None:
        movement_end = t0 + D  # end time of each movement
        t = np.linspace(min(t0), max(movement_end), num=100)
    
    # Initialize velocities
    vel = np.zeros((n_submovements, t.size))
    
    # Using minimum jerk, find velocity curve for each submovement
    for i_sub in range(n_submovements):
        vel[i_sub, :] = _minimum_jerk_velocity_1D(t0[i_sub], D[i_sub], A[i_sub], t)
    
    # Get total velocity expected from submovements
    sum_vel = np.sum(vel, axis=0)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot individual submovements
    vel_lines = []
    if plot_type == 1 or plot_type == 2:
        vel_lines = ax.plot(t, vel.transpose(), 'b', label='Submovements')
    
    # If plot_type is 1, also plot the sum of velocities
    if plot_type == 1 or plot_type == 3:
        vel_sum_line = ax.plot(t, sum_vel, 'r--', linewidth=2, label='Sum')
        if plot_type == 1:
            ax.legend(handles=[vel_lines[0], vel_sum_line[0]])
    else:
        vel_sum_line = None
        ax.legend(handles=[vel_lines[0]])
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity')
    ax.set_title('1D Submovements')
    ax.grid(True)
    
    return ax, fig, vel_lines, vel_sum_line