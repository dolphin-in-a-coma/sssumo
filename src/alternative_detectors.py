# Redefining the function here for execution
# %%

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt, cheby1
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import torch

try:
    from pybads import BADS 
except ImportError:
    pass # BADS = None


# %%

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

def minimal_jerk_trajectory(duration, amplitude=1):
    """
    Generate the minimal jerk trajectory for a submovement.
    """

    if duration - round(duration) != 0:
        raise NotImplementedError("Fractional durations are not supported yet.")
    t_normalized = np.linspace(1/duration, 1-1/duration, duration)
    velocity = t_normalized ** 2 * (1 - t_normalized) ** 2
    velocity /= np.sum(velocity)  # Normalize the velocity profile
    return velocity * amplitude

# def beta_shape_continuous(t, D, T0, T1, alpha=3, beta=3):
#     duration = T1
#     T1 += duration
#     t = np.clip(t, T0, T1) - T0
#     shape = (t / duration) ** (alpha - 1) * (1 - t / duration) ** (beta - 1)
#     return shape
    

def reconstruct_minimal_jerk_signal(submovements, N, return_mask=False):
    """
    Reconstruct the signal using minimal jerk trajectories for each submovement.
    
    Args:
    submovements (list): List of detected submovements (onset, peak, offset, etc.).
    N (int): Length of the original signal.
    
    Returns:
    numpy array: Reconstructed signal.
    """
    reconstructed_signal = np.zeros(N)

    if return_mask:
        mask_array = np.zeros(N)
        amplitude_array = np.zeros(N)
        duration_array = np.zeros(N)
    
    for submovement in submovements:
        # onset = submovement['onset']
        peak = submovement['peak']
        # offset = submovement['offset']
        duration = submovement['duration'] 
        amplitude = submovement['amplitude']

        # Reconstruct the minimal jerk trajectory for this submovement
        if duration % 2 != 0: # if duration is odd (in segments), it means that the peak is in the middle of the segment
            peak += 0.5
        segment_start = int(round(peak - duration/2))
        segment_end = int(round(peak + duration/2))
        segment_start_adjusted = max(0, segment_start)
        segment_end_adjusted = min(N, segment_end)

        submovement_shape = minimal_jerk_trajectory(duration)
        submovement_shape = submovement_shape[
            segment_start_adjusted-segment_start:
            len(submovement_shape) - (segment_end - segment_end_adjusted)]


        reconstructed_signal[segment_start_adjusted:segment_end_adjusted] += submovement_shape * amplitude

        if return_mask:
            mask_array[segment_start_adjusted] = 1
            amplitude_array[segment_start_adjusted] = amplitude
            duration_array[segment_start_adjusted] = duration

    if return_mask:
        mask_stacked = np.stack([mask_array, amplitude_array, duration_array], axis=1)
        return reconstructed_signal, mask_stacked
    else:
        return reconstructed_signal

def heuristic_submovement_detection(v, theta=0.05, smooth_method='gaussian', **kwargs):
    """
    Heuristic method to identify submovements from the input signal.
    
    Args:
    v (numpy array): Input signal v(t) for t = 1, 2, ..., N.
    theta (float): Threshold for identifying submovements.
    sigma (float): Smoothing parameter for the Gaussian filter.
    
    Returns:
    submovements (list): List of submovements, each represented by a dictionary with onset, peak, offset, duration, and amplitude.
    """
    
    # Step 0: Smooth the input signal
    if smooth_method == 'gaussian':
        sigma = kwargs.get('sigma', 2)
        v_smooth = gaussian_filter1d(v, sigma)
    elif smooth_method == 'cheby':
        fs = kwargs.get('fs', 60)
        cutoff = kwargs.get('cutoff', 5)
        order = kwargs.get('order', 4)
        ripple = kwargs.get('ripple', 0.1)
        b, a = cheby1(order, ripple, cutoff / (0.5 * fs), btype='low', analog=False)
        # b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)
        # Apply zero-phase filtering
        v_smooth = filtfilt(b, a, v)
    elif smooth_method == 'spline':
        s = kwargs.get('s', 0.15)
        k = kwargs.get('k', 2)
        x = np.arange(len(v))
        spline_function = UnivariateSpline(x, v, s=s, k=k)
        v_smooth = spline_function(x)
    else:
        raise ValueError(f"Invalid smooth_method: {smooth_method}")
    # Initialize empty lists
    peaks = []
    onset_candidates = [0]
    onset_candidates_label = []
    submovements = []

    is_stable = False
    
    # Step 1: Identify absolute peaks and valley extremes
    N = len(v)
    for t in range(1, N - 1):
        # Check for local maxima
        curr_sample = v_smooth[t]
        # that wasn't necessary
        prev_sample = v_smooth[t-1] # if t > 0 else v_smooth[t]
        next_sample = v_smooth[t+1] # if t < N-1 else v_smooth[t]

        if prev_sample <= curr_sample > next_sample:
            if curr_sample > theta:
                peaks.append(t)
            elif curr_sample < 0:
                onset_candidates.append(t)
                onset_candidates_label.append('negative maxima')
        
        # Check for local minima
        elif prev_sample >= curr_sample < next_sample:
            if curr_sample < -theta:
                peaks.append(t)
            elif curr_sample > 0:
                onset_candidates.append(t)
                onset_candidates_label.append('positive minima')

    # Step 2: Identify zero-crossing events and add to onset candidates
        if prev_sample * next_sample < 0:
            if len(onset_candidates) == 0 or onset_candidates[-1] != t:
                onset_candidates.append(t)
                onset_candidates_label.append('zero-crossing')
    
    # Step 3: Identify signal stability events and add to onset candidates

        if abs(curr_sample) < theta:
            if not is_stable:
                if len(onset_candidates) == 0 or onset_candidates[-1] != t:
                    onset_candidates.append(t)  # First stable time point
                    onset_candidates_label.append('stable begin')
                is_stable = True
        else:
            if is_stable:
                if len(onset_candidates) == 0 or onset_candidates[-1] != t-1:
                    onset_candidates.append(t-1)  # Last stable time point
                    onset_candidates_label.append('stable end')
                is_stable = False

    onset_candidates.append(N-1)
    # Step 4: Offset candidates are the same as onset candidates

    # Step 5: For each submovement peak, find the closest onset and offset candidates

    onset_idx = 0
    offset_idx = 0

    for peak_idx, peak in enumerate(peaks):
        # Find the closest onset and offset
        while onset_candidates[offset_idx] < peak:
            offset_idx += 1

        onset_idx = offset_idx - 1
        onset = onset_candidates[onset_idx]
        offset = onset_candidates[offset_idx] # if offset_idx < len(onset_candidates) else N-1
        
        # Calculate duration
        duration = offset - onset
        
        # Calculate amplitude (AUC)
        amplitude = np.sum(v[onset:offset+1])
        
        # Append submovement info
        submovements.append({
            'onset': onset,
            'peak': peak,
            'offset': offset,
            'duration': duration,
            'amplitude': amplitude
        })
    
    return submovements

def heuristic_detect_submovements_return_mask(v, theta=0.3, smooth_method='gaussian', **kwargs):
    """
    ...
    """
    submovements = heuristic_submovement_detection(v, theta, smooth_method, **kwargs)


    mask_array = np.zeros_like(v)
    amplitude_array = np.zeros_like(v)
    duration_array = np.zeros_like(v)

    for submovement in submovements:
        # onset = submovement['onset']
        peak = submovement['peak']
        # offset = submovement['offset']
        duration = submovement['duration'] 
        amplitude = submovement['amplitude']

        # Reconstruct the minimal jerk trajectory for this submovement
        if duration % 2 != 0: # if duration is odd (in segments), it means that the peak is in the middle of the segment
            peak += 0.5
        segment_start = int(round(peak - duration/2))
        # segment_end = int(round(peak + duration/2))
        segment_start_adjusted = max(0, segment_start)
        # segment_end_adjusted = min(N, segment_end)

        mask_array[segment_start_adjusted] = 1
        amplitude_array[segment_start_adjusted] = amplitude
        duration_array[segment_start_adjusted] = duration

    amplitude_array = fill_zeros_with_nearest_nonzero(amplitude_array)
    duration_array = fill_zeros_with_nearest_nonzero(duration_array)

    mask_stacked = np.stack([mask_array, amplitude_array, duration_array], axis=0)
    
    return mask_stacked

def hogan_optimize_submovements(v_t, t=None, solver='L-BFGS-B', max_submovements=10, error_threshold=0.02, num_starts=10, exact_number_of_submovements=False, fixed_initial_params=None, debug=False):
    """
    ...
    """

    NUM_PARAMS_PER_SUBMOVEMENT = 3




    if t is None:
        t = np.arange(len(v_t))


    fit = lambda x: hogan_fit_error([], v_t, t)
    best_error = fit([])
    best_params = []

    BOUND = [(min(t), max(t)), 
             (-30, 30),
             (5, 60)]
    
    if exact_number_of_submovements:
        n_submovements = max_submovements 
    else:
        if fixed_initial_params is not None:
            n_submovements = len(fixed_initial_params) // NUM_PARAMS_PER_SUBMOVEMENT
            assert n_submovements <= max_submovements, "Number of fixed submovements exceeds the maximum allowed."
        else:
            n_submovements = 1

    best_error_per_number_of_submovements = [best_error] + [np.inf] * max_submovements

    while best_error > error_threshold and n_submovements <= max_submovements:

        bounds = BOUND * n_submovements

        for i in range(num_starts):

            # Increase the number of submovements
            # Initialize random parameters for all submovements (5 parameters per submovement)
            initial_params = np.random.uniform(0, 1, size=(n_submovements * NUM_PARAMS_PER_SUBMOVEMENT))
            
            if fixed_initial_params is not None:
                initial_params[:len(fixed_initial_params)] = fixed_initial_params

            # print(initial_params)
            initial_params[::3] *= (BOUND[0][1] - BOUND[0][0])
            initial_params[::3] += BOUND[0][0]
            initial_params[1::3] *= (BOUND[1][1] - BOUND[1][0])
            initial_params[1::3] += BOUND[1][0]
            initial_params[2::3] *= (BOUND[2][1] - BOUND[2][0])
            initial_params[2::3] += BOUND[2][0]

            # Bounds: T0 < T1 and other constraints depending on physical limits
            

            # print(initial_params, bounds)
            
            # Perform local optimization using scipy.optimize.minimize (similar to fmincon)
            # res = minimize(fit_error, initial_params, args=(G_t, t), bounds=bounds, method='COBYLA')
            if solver == 'BADS':
                bads = BADS(lambda x: hogan_fit_error(x, v_t, t), 
                            initial_params, lower_bounds=[b[0] for b in bounds], upper_bounds=[b[1] for b in bounds])
                res = bads.optimize()
                error = res.fval
            else:
                res = minimize(hogan_fit_error, initial_params, args=(v_t, t), bounds=bounds, method=solver)
                error = res.fun
            
            params = res.x

            #print(optimize_result)

            # print(res.fun)

            t0 = params[::3]
            sorted_indices = np.argsort(t0)
            t0 = t0[sorted_indices]
            t0 = np.round(t0 + 0.5)
            t0_diff = np.diff(t0)
            if np.sum(t0_diff < 2) > 0:
                for i, t0_diff_el in enumerate(t0_diff):
                    if t0_diff_el < 2:
                        params[sorted_indices[i]*3] -= 1
                        params[sorted_indices[i+1]*3] += 1

                error = hogan_fit_error(params, v_t, t)
                        # move submovements

            if error < best_error_per_number_of_submovements[n_submovements]:
                best_error_per_number_of_submovements[n_submovements] = error

            # Check if this is the best solution so far
            if error < best_error:
                prev_best_error = best_error
                best_error = error
                best_params = res.x

                t0 = best_params[::3]
                t0 = np.round(t0 + 0.5)
                t0 = sorted(t0)
                t0_diff = np.diff(t0)
                if len(t0_diff) > 0 and np.all(t0_diff >= 3):
                    best_error = prev_best_error
        
        increase_latency = 2
        first_submovement2check = max(0, n_submovements - increase_latency)
        last_submovement2check = max(0, n_submovements - 1)
        if best_error_per_number_of_submovements[n_submovements] >= max(best_error_per_number_of_submovements[first_submovement2check:last_submovement2check+1]):
            if debug:
                print(f'No improvement for {increase_latency} submovements, stopping the optimization.')
            break
        n_submovements += 1
            # bounds.extend(BOUND)

    if best_params is None:
        return [], np.inf
        
        # Return the best found parameters and the corresponding error
    return best_params, best_error


def hogan_detect_submovements_return_mask(v, mask=None, solver='L-BFGS-B', max_submovements=12, error_threshold=0.05, num_starts=10, overlap=6, debug=False):
    """
    ...
    """

    v_abs_mean = np.mean(np.abs(v))

    if overlap is None:
        overlap = max_submovements - 5

    if mask is not None:
        submovement_onsets = np.where(mask[0] == 1)[0]
        total_number_of_submovements = len(submovement_onsets)
    else:
        AVERAGE_SUBMOVEMENT_REFRACTORY_TIME = 5
        submovement_onsets = np.arange(0, v.shape[0], AVERAGE_SUBMOVEMENT_REFRACTORY_TIME)
        total_number_of_submovements = len(submovement_onsets)
    
        
    all_submovement_params = []

    first_submovement = 0

    while first_submovement < total_number_of_submovements - overlap:
        after_last_submovement_no_overlap = min(first_submovement + max_submovements - overlap, total_number_of_submovements)
        after_last_submovement = min(first_submovement + max_submovements, total_number_of_submovements)
        segment_start = submovement_onsets[first_submovement]
        segment_end = submovement_onsets[after_last_submovement] if after_last_submovement < total_number_of_submovements else v.shape[0]
        segment_end_no_overlap = submovement_onsets[after_last_submovement_no_overlap] if after_last_submovement_no_overlap < total_number_of_submovements else v.shape[0]
        v_segment = v[segment_start:segment_end]
        number_submovements = after_last_submovement - first_submovement
        # number_submovements_no_overlap = after_last_submovement_no_overlap - first_submovement

        error_threshold_adjusted = error_threshold * v_abs_mean / np.mean(np.abs(v_segment))
 
        submovement_params, error = hogan_optimize_submovements(v_segment,
                                                            solver=solver,
                                                            max_submovements=number_submovements,
                                                            error_threshold=error_threshold_adjusted,
                                                            num_starts=num_starts,
                                                            exact_number_of_submovements=False,
                                                            debug=debug)

        if len(submovement_params) > 0:
            submovement_params_grouped = np.split(submovement_params, len(submovement_params) // 3)
            submovement_params_grouped = np.stack(submovement_params_grouped, axis=0)
            # submovement_params_grouped[:, 0] += segment_start
            submovement_params_grouped[:, 0] = np.round(submovement_params_grouped[:, 0] + 0.5)
            t0 = submovement_params_grouped[:, 0]
            if first_submovement < total_number_of_submovements - overlap * 2:
                submovement_params_grouped = submovement_params_grouped[ t0 < segment_end_no_overlap - segment_start]
            submovement_params = submovement_params_grouped.flatten()

        # submovement_params = submovement_params[:number_submovements_no_overlap*3]
        

        reconstructed_segment_length = segment_end_no_overlap - segment_start + 60 # max duration of a submovement
        reconstructed_segment_length = min(reconstructed_segment_length, v.shape[0] - segment_start)

        _, v_segment_reconstructed = hogan_fit_error(submovement_params,
                                                     np.zeros(reconstructed_segment_length),
                                                     np.arange(reconstructed_segment_length),
                                                     return_reconstructed=True)
        
        
        submovement_params[::3] += segment_start
        all_submovement_params.extend(submovement_params)

        if debug:
            plt.figure(dpi=150)
            plt.plot(v[segment_start:segment_end], label='Velocity segment')
            plt.plot(v_segment_reconstructed, label='Reconstructed velocity segment')


        v[segment_start:segment_start+reconstructed_segment_length] -= v_segment_reconstructed

        if debug:
            plt.plot(v[segment_start:segment_end], label='Velocity residual')

            plt.legend()
            plt.xlabel('Time, samples')
            plt.ylabel('Velocity, units/samples')
            plt.title(f'Segments prediction and residuals, error: {error}, submovements number: {len(submovement_params)//3}')
            plt.show()

        if debug:
            print(f'\tSegment: {first_submovement}-{after_last_submovement-1} / {total_number_of_submovements} processed.')
        
        first_submovement += max_submovements - overlap

    # submovements = []
    mask_array = np.zeros_like(v)
    amplitude_array = np.zeros_like(v)
    duration_array = np.zeros_like(v)

    for i in range(0, len(all_submovement_params), 3):
        onset = all_submovement_params[i]
        onset = round(onset)
        amplitude = all_submovement_params[i+1]
        duration = all_submovement_params[i+2]

        # submovements.append({
        #     'onset': onset,
        #     # 'peak': onset + duration/2,
        #     # 'offset': onset + duration,
        #     'duration': duration,
        #     'amplitude': amplitude
        # })

        mask_array[onset] = 1
        amplitude_array[onset] = amplitude
        duration_array[onset] = duration

    if debug:
        print(all_submovement_params)

    amplitude_array = fill_zeros_with_nearest_nonzero(amplitude_array)
    duration_array = fill_zeros_with_nearest_nonzero(duration_array)

    mask_stacked = np.stack([mask_array, amplitude_array, duration_array], axis=0)
    
    return mask_stacked

def minimal_jerk_trajectory_continuous(t, D, T0, duration):
    """
    Generate the minimal jerk trajectory for a submovement.
    """

    T1 = T0 + duration
    t_normalized = (t - T0) / duration
    t_normalized = np.clip(t_normalized, 0, 1)
    shape = D / duration * (30 * t_normalized ** 2 - 60 * t_normalized ** 3 + 30 * t_normalized ** 4)
    shape *= D/sum(shape)
    return shape

def LGNB(t, D, T0, duration, mu=0, sigma=0.8):
    # t = np.clip(t, T0, T1)
    # print(t)
    # LGNB formula implementation

    T1 = T0 + duration
    shape = D * (T1 - T0) / (sigma * np.sqrt(2 * np.pi) * (t - T0) * (T1 - t)) * np.exp(
        -0.5 * ((np.log((t - T0) / (T1 - t)) - mu) / sigma) ** 2
    )
    shape = np.nan_to_num(shape, 0)
    return shape

def hogan_fit_error(params, G_t, t, return_reconstructed=False):
    # Number of submovements (based on the size of parameters)
    PARAMETERS_PER_SUBMOVEMENT = 3
    n_submovements = len(params) // PARAMETERS_PER_SUBMOVEMENT
    
    # Initialize the extracted speed profile (F_t)
    F_t = np.zeros_like(G_t)
    
    # Sum up all submovement contributions
    for i in range(n_submovements):
        T0, D, duration = params[i*PARAMETERS_PER_SUBMOVEMENT:
                                 (i+1)*PARAMETERS_PER_SUBMOVEMENT]
        F_t += minimal_jerk_trajectory_continuous(t, D, T0, duration)

    # plt.figure()
    # plt.plot(F_t)
    # plt.show()
    
    # Fit error (E) based on equation (2)
    error = np.sum(np.abs(F_t - G_t)) / np.sum(np.abs(G_t))

    # print(error, n_submovements)

    if return_reconstructed:
        return error, F_t
    else:
        return error
    
def heuristic_pytorch_style(x, theta=0.1, smooth_method='gaussian', **kwargs):

    x = x.squeeze(1) # remove the channel dimension

    y_pred = []
    for i, x_el in enumerate(x):
        y_el_pred = heuristic_detect_submovements_return_mask(x_el.numpy(), theta=theta, smooth_method=smooth_method, **kwargs)
        y_pred.append(y_el_pred)

    y_pred = np.stack(y_pred, axis=0)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    return y_pred

def hogan_pytorch_style(x, mask_true=None, solver='L-BFGS-B', max_submovements=10, error_threshold=0.05, num_starts=10, debug=False):
    """
    """

    x = x.squeeze(1) # remove the channel dimension

    y_pred = []
    for i, x_el in enumerate(x):
        if debug:
            print(f'Processing batch element {i+1}/{len(x)}')

        if mask_true is not None:
            mask_true_el = mask_true[i].numpy()
        else:
            mask_true_el = None
        y_el_pred = hogan_detect_submovements_return_mask(x_el.numpy().copy(),
                                                          mask_true_el,
                                                          solver=solver,
                                                          max_submovements=max_submovements,
                                                          error_threshold=error_threshold,
                                                          num_starts=num_starts,
                                                          debug=debug)
        y_pred.append(y_el_pred)

    y_pred = np.stack(y_pred, axis=0)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    return y_pred