# %%

from abc import ABC, abstractmethod

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt

import fastkde
import torch
from torch.utils.data import Dataset, DataLoader

from torch.distributions import Categorical, Normal, Uniform, Distribution, Chi2

import random

from collections.abc import Iterable

from models import Reconstructor, STEContinuousReconstructor

import torch
from torch.distributions import LogNormal, constraints
from torch.distributions.utils import broadcast_all


DTYPE = torch.float32
DEVICE = 'cpu'
BASE_RECONSTRUCTOR = STEContinuousReconstructor

BASE_RECONSTRUCTOR_PARAMS = parameters_dict = {
    "duration_range": (5, 60),
    "freeze_primitive_parameters": True,
    "primitive_beta_mean": (0.5, 0.0),
    "primitive_beta_precision": (6.0, 0.0),
    'device': DEVICE,
    'dtype': DTYPE,
}


# %%

# %%

class Sampler(ABC):
    @abstractmethod
    def __init__(self, data_df=None, pdf=None, x_label=None, h_y=None):
        pass
    @abstractmethod
    def sample(self, add_noise=True):
        pass
    
    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def _estimate_distribution(self, data_df):
        pass

    @abstractmethod
    def _precompute_cdf(self):
        pass

class Sampler1D(Sampler):
    def __init__(self, pdf=None, data_df=None, x_label=None, h_y=None):

        self.x_label = x_label
        
        if pdf is None:
            pdf = self._estimate_distribution(data_df)
        self.pdf = pdf

        if self.x_label is None:
            self.x_label = self.pdf.coords.dims[0]

        self.x_grid = self.pdf.coords[self.x_label].values

        self._interpolate_nans()
        self.cdf = self._precompute_cdf()

        self.h_y = h_y if h_y is not None else np.diff(self.x_grid).mean()

    def _estimate_distribution(self, data_df):
        if self.x_label is None:
            print("x_label is not specified, using first column")
            self.x_label = data_df.columns[0]

        pdf = fastkde.pdf(data_df[self.x_label].values, var_names=[self.x_label])
        return pdf
    
    def _interpolate_nans(self):
        self.pdf = self.pdf.interpolate_na(dim=self.x_label, method="zero", fill_value="extrapolate")

    def _precompute_cdf(self):
        pdf_normalized = self.pdf / np.trapz(self.pdf, self.x_grid)
        pdf_normalized = pdf_normalized.values
        # Compute CDF
        dx = np.diff(self.x_grid)
        # print(pdf_normalized.shape, dx.shape)
        area = 0.5 * (pdf_normalized[:-1] + pdf_normalized[1:]) * dx
        cdf = np.zeros_like(self.x_grid)
        cdf[1:] = np.cumsum(area)
        cdf /= cdf[-1]  # Ensure CDF ends at 1
        return cdf
    
    def plot(self):
        plt.figure(figsize=(8, 4))
        plt.subplot(2, 1, 1)
        plt.plot(self.x_grid, self.pdf, label='PDF')
        plt.xlabel(self.x_label)
        plt.ylabel('PDF')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(self.x_grid, self.cdf, label='CDF')
        plt.xlabel(self.x_label)
        plt.ylabel('CDF')
        plt.legend()
        plt.show()

    def sample(self, n_samples=1, add_noise=True):
        u = np.random.rand(n_samples)
        samples = np.interp(u, self.cdf, self.x_grid)
        if add_noise:
            samples += np.random.randn(n_samples) * self.h_y
        return samples

class ConditionalSampler(Sampler):
    def __init__(self, data_df=None, cPDF=None, x_label=None, y_label=None, h_y=None):
        """
        Initialize the sampler with the conditional PDF (PDF(Y|X=x)).
        
        Args:
            cPDF (xarray.DataArray): Conditional PDF with dimensions (y, x).
            h_y (float): Bandwidth for Y (for adding KDE noise). If None, use grid spacing.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
        """
        self.x_label = x_label
        self.y_label = y_label

        if cPDF is None:
            cPDF = self._estimate_distribution(data_df)
        self.cPDF = cPDF

        if self.x_label is None:
            self.x_label = self.cPDF.coords.dims[1]
        if self.y_label is None:
            self.y_label = self.cPDF.coords.dims[0]
        
        self.x_grid = self.cPDF.coords[self.x_label].values
        self.y_grid = self.cPDF.coords[self.y_label].values

        self._interpolate_nans()
        # Precompute CDFs for all x-grid points (for fast interpolation)
        self.cdfs = self._precompute_cdf()
        
        # Bandwidth for Y (KDE smoothing)
        self.h_y = h_y if h_y is not None else np.diff(self.y_grid).mean()

        self._mean = None

    def _estimate_distribution(self, data_df):

        if self.x_label is None:
            print("x_label is not specified, using first column")
            self.x_label = data_df.columns[0]
        if self.y_label is None:
            print("y_label is not specified, using second column")
            self.y_label = data_df.columns[1]

        cPDF = fastkde.conditional(
            [data_df[self.y_label].values], 
            [data_df[self.x_label].values], 
            var_names=[self.x_label, self.y_label]
        )
        return cPDF

    def _interpolate_nans(self):
        """Interpolate NaNs in the conditional PDF."""
        self.cPDF = self.cPDF.interpolate_na(dim=self.x_label, method="nearest", fill_value="extrapolate")

    def _precompute_cdf(self):
        """Precompute CDFs for all x-grid points using trapezoidal integration."""
        cdfs = []
        for x_idx in range(len(self.x_grid)):
            pdf = self.cPDF.isel({self.x_label: x_idx}).values
            pdf_normalized = pdf / np.trapz(pdf, self.y_grid)
            
            # Compute CDF
            dx = np.diff(self.y_grid)
            area = 0.5 * (pdf_normalized[:-1] + pdf_normalized[1:]) * dx
            cdf = np.zeros_like(self.y_grid)
            cdf[1:] = np.cumsum(area)
            cdf /= cdf[-1]  # Ensure CDF ends at 1
            cdfs.append(cdf)
        
        return np.array(cdfs)

    def plot(self):
        plt.figure(figsize=(8, 4))
        plt.subplot(2, 1, 1)
        plt.imshow(self.cPDF.values, extent=[self.x_grid.min(), self.x_grid.max(), self.y_grid.min(), self.y_grid.max()], origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(label=f'PDF({self.y_label}|{self.x_label}={self.x_label}_0)')
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title('Conditional PDF')
        
        plt.subplot(2, 1, 2)
        plt.imshow(self.cdfs.T, extent=[self.x_grid.min(), self.x_grid.max(), self.y_grid.min(), self.y_grid.max()], origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(label=f'CDF({self.y_label}|{self.x_label}={self.x_label}_0)')
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title('Conditional CDFs')
        plt.show()
        
    def sample(self, x0, add_noise=True):
        """
        Sample Y values given an array of x0 values (one sample per x0).
        
        Args:
            x0 (np.ndarray): Array of X values to condition on.
            add_noise (bool): Add KDE bandwidth noise to samples.
            
        Returns:
            np.ndarray: Sampled Y values (shape matching x0).
        """
        # Clamp x0 to the grid boundaries

        x0_shape = x0.shape
        x0 = x0.flatten()

        x0_clipped = np.clip(x0, self.x_grid.min(), self.x_grid.max())
        
        # Find nearest left/right x indices and interpolation weights
        idx_right = np.searchsorted(self.x_grid, x0_clipped, side='right')
        idx_left = np.maximum(idx_right - 1, 0)
        idx_right = np.minimum(idx_right, len(self.x_grid) - 1)
        
        # Compute interpolation weights
        x_left = self.x_grid[idx_left]
        x_right = self.x_grid[idx_right]
        eps = 1e-10  # Avoid division by zero
        weight_right = (x0_clipped - x_left) / (x_right - x_left + eps)
        weight_left = 1 - weight_right
        
        # Linearly interpolate between neighboring CDFs
        cdf_interp = (
            weight_left[:, None] * self.cdfs[idx_left] + 
            weight_right[:, None] * self.cdfs[idx_right]
        )
        
        # Inverse transform sampling
        u = np.random.rand(len(x0))
        samples = np.array([
            np.interp(u[i], cdf_interp[i], self.y_grid) 
            for i in range(len(x0))
        ])
        
        # Add KDE bandwidth noise
        if add_noise:
            samples += np.random.randn(len(samples)) * self.h_y

        samples = samples.reshape(x0_shape)
        
        return samples
    
    def chained_sample(self, x0, n_samples=1, add_noise=True):
        samples = []
        x0 = np.array(x0)
        if x0.ndim == 0:
            x0 = x0.reshape(-1)

        samples_matrix = np.zeros((n_samples + 1, len(x0)))
        samples_matrix[0] = x0
        
        x_i = x0
        for i in range(n_samples):
            x_i = self.sample(x0=x_i, add_noise=add_noise)
            samples_matrix[i+1] = x_i

        samples_matrix = samples_matrix.T
        return samples_matrix
    
    # def mean(self, x_pdf=None):
    #     if self._mean is not None:
    #         return self._mean
        
    #     assert x_pdf is not None, "x_pdf is not specified"
    #     # self.cPDF
        
class ConstantDistribution(Distribution):
    def __init__(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self.value = value

    def sample(self, sample_shape=torch.Size()):
        # Return the constant value
        return self.value.expand(sample_shape)

    def log_prob(self, value):
        # Log probability is 0 if value matches, otherwise -inf
        return torch.where(value == self.value, torch.tensor(0.0), torch.tensor(float('-inf')))
    
    def mean(self):
        return self.value
    
class TruncatedLogNormal(Distribution):
    """
    LogNormal distribution truncated to the interval [a, b].

    Args:
        loc (float or Tensor): Mean of the underlying Gaussian distribution (often denoted mu).
        scale (float or Tensor): Standard deviation of the underlying Gaussian distribution (often denoted sigma).
        a (float or Tensor): Lower truncation bound. Must be positive.
        b (float or Tensor): Upper truncation bound. Must be greater than a.
        validate_args (bool, optional): Whether to validate arguments. Default None.
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive,
                       'a': constraints.positive, 'b': constraints.positive}
    # Although support is technically [a, b], defining it dynamically is complex.
    # constraints.positive is inherited from LogNormal and broadly correct.
    support = constraints.positive 
    has_rsample = False # ICDF approach used in sample might not be reparameterizable

    def __init__(self, loc=3., scale=0.6, a=5, b=10, validate_args=None):
        # Ensure a and b are tensors and broadcast them with loc and scale
        _loc, _scale, _a, _b = broadcast_all(loc, scale, a, b)
        
        # LogNormal requires positive scale
        # 'a' must be positive for LogNormal support
        # Clamp 'a' slightly above zero if needed for numerical stability
        _a = torch.clamp(_a, min=torch.finfo(_loc.dtype).eps) 

        if validate_args:
             if torch.any(_a <= 0):
                 raise ValueError("Truncation lower bound 'a' must be positive.")
             if torch.any(_a >= _b):
                 raise ValueError("Truncation lower bound 'a' must be less than upper bound 'b'.")

        self.loc, self.scale, self.a, self.b = _loc, _scale, _a, _b
        
        super().__init__(batch_shape=self.loc.shape, event_shape=torch.Size(), validate_args=validate_args)

        self.base_dist = LogNormal(self.loc, self.scale)
        
        # Calculate CDF values at bounds for normalization and sampling
        cdf_a = self.base_dist.cdf(self.a)
        cdf_b = self.base_dist.cdf(self.b)
        
        # Clamp normalization factor to avoid log(0) or division by zero.
        self.normalization = (cdf_b - cdf_a).clamp(min=torch.finfo(self.loc.dtype).eps)
        self.log_normalization = self.normalization.log()
        
        # Store CDF values needed for sampling
        self._cdf_a = cdf_a

        self.low = self.a
        self.high = self.b

        # self._cdf_b = cdf_b # Not strictly needed if self.normalization and self._cdf_a are stored

    @property
    def mean(self):
        """
        Calculates the mean of the truncated LogNormal distribution.
        mean = E[X | a <= X <= b] 
             = exp(mu + sigma^2/2) * [Phi( (ln(b) - mu - sigma^2) / sigma ) - Phi( (ln(a) - mu - sigma^2) / sigma )] / [Phi( (ln(b) - mu) / sigma ) - Phi( (ln(a) - mu) / sigma )]
        Where Phi is the CDF of the standard normal distribution N(0, 1).
        """
        std_normal = Normal(torch.zeros_like(self.loc), torch.ones_like(self.scale))
        
        # Use stored self.a, self.b which might have been clamped
        log_a = torch.log(self.a) 
        log_b = torch.log(self.b)
        
        # Parameters for the denominator CDFs: (ln(x) - mu) / sigma
        alpha = (log_a - self.loc) / self.scale
        beta = (log_b - self.loc) / self.scale
        
        # Parameters for the numerator CDFs: (ln(x) - mu - sigma^2) / sigma
        alpha_prime = alpha - self.scale # More stable than recalculating
        beta_prime = beta - self.scale  # More stable than recalculating
        # alpha_prime = (log_a - self.loc - self.scale**2) / self.scale # Original formulation
        # beta_prime = (log_b - self.loc - self.scale**2) / self.scale  # Original formulation

        # Denominator: Phi(beta) - Phi(alpha) = P(a <= X <= b)
        # We use the pre-calculated self.normalization which is (cdf_b - cdf_a)
        prob_mass = self.normalization 
        
        # Numerator term: Phi(beta') - Phi(alpha')
        cdf_alpha_prime = std_normal.cdf(alpha_prime)
        cdf_beta_prime = std_normal.cdf(beta_prime)
        num_term = cdf_beta_prime - cdf_alpha_prime

        # Factor exp(mu + sigma^2 / 2)
        mean_factor = torch.exp(self.loc + 0.5 * self.scale**2)
        
        # Calculate mean: factor * (num_term / prob_mass)
        mean_val = mean_factor * (num_term / prob_mass)
        
        # Handle cases where prob_mass was zero (e.g., b <= a). Return NaN.
        mean_val = torch.where(prob_mass <= torch.finfo(self.loc.dtype).eps, 
                               torch.full_like(mean_val, float('nan')), 
                               mean_val)
                               
        # The mean should fall within [a, b], but numerical issues are possible.
        # Clamping can hide issues, so we don't clamp by default.
        # mean_val = torch.clamp(mean_val, min=self.a, max=self.b) 

        return mean_val

    def sample(self, sample_shape=torch.Size()):
        """
        Samples from the truncated LogNormal distribution using inverse transform sampling.
        """
        shape = self._extended_shape(sample_shape)
        # Sample uniform values in the range [CDF(a), CDF(b)]
        # Use device and dtype of parameters
        u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device) 
        u = self._cdf_a + u * self.normalization # Scale and shift u to [cdf_a, cdf_b]
        
        # Ensure u stays within the valid range for icdf, avoiding numerical issues at boundaries
        u = u.clamp(min=torch.finfo(u.dtype).eps, max=1.0 - torch.finfo(u.dtype).eps)

        # Apply inverse CDF (percent point function) of the base LogNormal distribution
        samples = self.base_dist.icdf(u)
        
        # Due to potential numerical inaccuracies in icdf near boundaries, 
        # explicitly clamp the result to the interval [a, b]. This ensures validity.
        samples = torch.clamp(samples, min=self.a, max=self.b) 
        return samples

    def log_prob(self, value):
        """
        Calculates the log probability density function (log PDF) of the truncated LogNormal distribution.
        log_prob = log(f(x) / (CDF(b) - CDF(a))) if a <= x <= b, else -inf
                 = base_dist.log_prob(x) - log_normalization if a <= x <= b, else -inf
        """
        if self._validate_args:
            self._validate_sample(value)
            
        # Calculate log_prob using the base distribution and subtract log_normalization
        log_p = self.base_dist.log_prob(value) - self.log_normalization
        
        # Set log_prob to -inf for values outside the truncation interval [a, b]
        mask = (value >= self.a) & (value <= self.b)
        log_p = torch.where(mask, log_p, torch.full_like(log_p, float('-inf')))
        
        return log_p

class ModulatedGaussian(Distribution):
    def __init__(self, mean=0, std=1, poly_order=1):
        super().__init__()
        mean, std, poly_order = broadcast_all(mean, std, poly_order)
        self._mean = mean
        self.std = std
        self.poly_order = poly_order
        # compute base-normal sigma so that overall Var=std^2
        factor = (2 * math.gamma((poly_order + 3) / 2)
                  / math.gamma((poly_order + 1) / 2))
        self.base_sigma = std * factor**-0.5


        # precompute log-normalizer:
        #   log C = -((k+1)/2)*log(2 σ^2) - log Γ((k+1)/2)
        self.log_C = -((poly_order + 1) / 2) * math.log(2 * self.base_sigma**2) \
                     - math.lgamma((poly_order + 1) / 2)
        
    @property
    def mean(self):
        return self._mean
    
    @mean.setter
    def mean(self, value):
        self._mean = value

    def log_prob(self, value):
        # value: Tensor of shape [...], same dtype/device as mean/std
        diff = value - self.mean
        # k * log|x-μ|
        term1 = self.poly_order * torch.log(torch.abs(diff))
        # Gaussian exponent
        term2 = -0.5 * (diff**2) / (self.base_sigma**2)
        return self.log_C + term1 + term2
    
    def sample(self, sample_shape=torch.Size()):
        # 1) Draw Y ~ |X−loc| by sampling a chi distribution:
        #    If Z ∼ Chi2(k+1), then |X−loc| = sqrt(Z) * base_sigma
        df = self.poly_order + 1
        z = Chi2(df).sample(sample_shape)
        y = torch.sqrt(z) * self.base_sigma

        # 2) Random sign for symmetry
        signs = torch.where(torch.rand_like(y) < 0.5, -1.0, 1.0)
        return self.mean + signs * y
class ConditionalDistribution(Distribution):
    def __init__(self, data_df, x_label=None, y_labels=None, next_x_label=None, add_noise=True, make_symmetric=False, values_require_symmetry=('curr_amplitude' , 'curr_mean_velocity', 'next_amplitude' , 'next_mean_velocity')):

        columns = data_df.columns
        if x_label is None:
            x_label = columns[0]
        if y_labels is None:
            y_labels = columns[1:]
            if next_x_label is not None:
                y_labels = [label for label in columns if label != next_x_label]
        
        # if len(y_labels) == 0:
        #     y_labels = None

        self.x_label = x_label
        self.y_labels = y_labels
        self.next_x_label = next_x_label
        data_df = self.filter_data(data_df)

        self.make_symmetric = make_symmetric
        self.values_require_symmetry = values_require_symmetry
        if make_symmetric:
            data_df = self.symmetrize_data(data_df)

        self.x_sampler = Sampler1D(data_df=data_df, x_label=x_label)
        
        self.y_samplers = [ConditionalSampler(data_df=data_df, x_label=x_label, y_label=y_label) for y_label in y_labels]

        if next_x_label is not None:
            self.next_x_sampler = ConditionalSampler(data_df=data_df, x_label=x_label, y_label=next_x_label) 
        else:
            self.next_x_sampler = None

        self.add_noise = add_noise

    def filter_data(self, data_df):
        return data_df
    
    def symmetrize_data(self, data_df):
        data_df_symmetric = data_df.copy()
        for column in self.values_require_symmetry:
            if column in data_df.columns:
                data_df_symmetric[column] = -data_df_symmetric[column]
        return pd.concat([data_df, data_df_symmetric])
    
    def sample_x(self, n_samples=1):
        return self.x_sampler.sample(n_samples=n_samples, add_noise=self.add_noise)
    
    def sample_y(self, x0):
        list_of_samples = []
        for y_sampler in self.y_samplers:
            list_of_samples.append(y_sampler.sample(x0, add_noise=self.add_noise))
        return list_of_samples
    
    def x_chained_sample(self, sample_dim, sample_length):
        assert self.next_x_sampler is not None, "next_x_sampler is not specified"
        x0 = self.sample_x(n_samples=sample_dim)
        samples = self.next_x_sampler.chained_sample(x0=x0, n_samples=sample_length-1)
        return samples


# %%
if __name__ == '__main__':
    path = 'steering-1d_inf-pulled_stats'
    df = pd.read_csv(path)
    x_label = 'curr_amplitude'
    y_labels = ['curr_duration', 'curr_refractory']
    next_x_label = 'next_amplitude'
    cond_dist = ConditionalDistribution(data_df=df, x_label=x_label, y_labels=y_labels, next_x_label=next_x_label)
    import time
    tm = time.time()
    for i in range(100):
        x_sampled = cond_dist.x_chained_sample(sample_dim=150, sample_length=20)
        y_sampled = cond_dist.sample_y(x_sampled)
    print(time.time() - tm)

# %%
# test sampler1d
if __name__ == '__main__':
    data_df = pd.DataFrame({'x': np.random.randn(100000)})
    sampler = Sampler1D(data_df=data_df)

    samples = sampler.sample(n_samples=10000)
    plt.hist(samples, bins=100)
    plt.show()

    data_df['y'] = -data_df['x'] + np.random.randn(100000) * 0.3

    sampler = ConditionalSampler(data_df=data_df, x_label='x', y_label='y')
    x0 = np.linspace(0, 1, 1000)
    samples = sampler.sample(x0=x0)
    plt.scatter(x0, samples)
    plt.show()

    samples = sampler.chained_sample(x0=[0, 1, 2], n_samples=100)
    plt.scatter(samples[:-1], samples[1:])
    plt.show()

# %%

def convert_distribution(distribution):
    if isinstance(distribution, (int, float)):
        return ConstantDistribution(distribution)
    elif isinstance(distribution, Iterable):
        return Uniform(*distribution)
    elif isinstance(distribution, Distribution):
        return distribution

def generate_labels_from_parameters(total_duration, duration_distribution, amplitude_distribution, refractory_distribution,
                                    max_submovements=None, dtype=DTYPE, seed=None, batch_size=10, device=DEVICE,
                                    ):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # total_duration_distribution = convert_distribution(total_duration_distribution)
    duration_distribution = convert_distribution(duration_distribution)
    amplitude_distribution = convert_distribution(amplitude_distribution)
    refractory_distribution = convert_distribution(refractory_distribution)
    # sign_distribution = RandomChoice([-1, 1])

    # snr_distribution = convert_distribution(snr_distribution)
    mean_refractory = refractory_distribution.mean
    if max_submovements is None:
        max_submovements = total_duration // mean_refractory
        max_submovements = max_submovements.int().item()

    refractories = refractory_distribution.sample([batch_size, max_submovements])
    # start_times = torch.cumsum(refractories, dim=0)
    durations = duration_distribution.sample([batch_size, max_submovements])
    amplitudes = amplitude_distribution.sample([batch_size, max_submovements])
    # signs = sign_distribution.sample([max_submovements])
    # amplitudes = signs * amplitudes

    condition1 = amplitudes[:, 1:] * amplitudes[:, :-1] < 0
    diff = durations[:, :-1] - (durations[:, 1:] + refractories[:, 1:])
    condition2 = diff > 0
    condition = condition1 & condition2

    if condition.sum() > 0:
        # I don't remember why I did this
        # probably shifting the refractory periods of submovements that are too close 
        shifts = refractory_distribution.sample(torch.Size([condition.sum().item()]))
        shifts -= shifts.min()
        refractories[:,1:][condition] += diff[condition] + shifts

    start_times = torch.cumsum(refractories, dim=1)

    return start_times, durations, amplitudes

# %%
# datasets

class AbstractDataset(Dataset, ABC):

    def __init__(self, dtype=DTYPE, device=DEVICE):
        self.snr2log_jitter_intercept = 0.5
        self.snr2log_jitter_slope = -0.17
        self.dtype = dtype
        self.device = device


    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def convert_distribution(self, distribution):
        if distribution is None:
            return None
        if isinstance(distribution, (int, float)):
            distribution = torch.tensor(distribution, dtype=self.dtype, device=self.device)
            return ConstantDistribution(distribution)
        elif isinstance(distribution, Iterable):
            if not isinstance(list(distribution)[0], str):
                distribution = [torch.tensor(d, dtype=self.dtype, device=self.device) for d in distribution]
                return Uniform(*distribution)
            else:
                if distribution[0] == 'TruncatedLogNormal':
                    return TruncatedLogNormal(*[torch.tensor(d, dtype=self.dtype, device=self.device) for d in distribution[1:]])
                elif distribution[0] == 'Gaussian':
                    return Normal(*[torch.tensor(d, dtype=self.dtype, device=self.device) for d in distribution[1:]])
                elif distribution[0] == 'ModulatedGaussian':
                    return ModulatedGaussian(*[torch.tensor(d, dtype=self.dtype, device=self.device) for d in distribution[1:]])
                else:
                    raise ValueError(f"Unknown distribution type: {distribution[0]}")
        elif isinstance(distribution, Distribution):
            if distribution.device != self.device:
                raise ValueError("Distribution device does not match dataset device, please specify device initializing the distribution")
            return distribution
        
    def snr2jitter_sigma(self, snr):
        snr_range = (-5, np.inf)

        if isinstance(snr, (int, float)):
            snr = np.clip(snr, *snr_range)
            log_jitter_sigma = self.snr2log_jitter_intercept + self.snr2log_jitter_slope * snr
            return 2**log_jitter_sigma
        elif isinstance(snr, torch.Tensor):
            snr = snr.clamp(-5)
            log_jitter_sigma = self.snr2log_jitter_intercept + self.snr2log_jitter_slope * snr
            return torch.exp2(log_jitter_sigma)
        else:
            raise ValueError("snrs should be a number or a tensor")
    
    @staticmethod
    def interpolate(y, x_new, x=None):
        if x is None:
            x = torch.arange(y.shape[-1]).expand(y.shape)
        x = x.int()
        x_new = torch.clamp(x_new, 0, y.shape[-1] - 1)
        x_new_left = torch.floor(x_new).to(torch.int64)
        x_new_right = torch.ceil(x_new).to(torch.int64)
        weight_left = x_new_right - x_new
        weight_right = x_new - x_new_left
        weight_left[x_new_left == x_new_right] = 1

        y_new_left = torch.gather(y, -1, x_new_left)
        y_new_right = torch.gather(y, -1, x_new_right)
        y_new = y_new_left * weight_left + y_new_right * weight_right
        return y_new

    def add_jitter_noise(self, signal, snr, mean_kernel_size=11):
        # Step 1: Create signal from signal_delta
        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0)
        # if len(signal.shape) == 2:
        #     signal = signal.unsqueeze(0)
        # if len(signal.shape) > 3:
        #     raise ValueError("Signal should have at most 3 dimensions")
        
        assert mean_kernel_size % 2 == 1, "mean_kernel_size should be odd"

        jitter_sigma = self.snr2jitter_sigma(snr)


        signal_cumsum = torch.cumsum(signal, dim=-1)

        # shift signal_cumsum right by 1
        # signal_cumsum = torch.roll(signal_cumsum, 1, dims=-1)
        # signal_cumsum[..., 0] = 0
        signal_cumsum = torch.cat([torch.zeros_like(signal_cumsum[..., :1]), signal_cumsum], dim=-1)

        # signal_cumsum = torch.stack([signal_cumsum] * 30)
        # signal_cumsum = signal_cumsum.unsqueeze(1)

        # Step 2: Jitter processing
        time = torch.arange(signal_cumsum.shape[-1]).unsqueeze(0).unsqueeze(0)
        if isinstance(jitter_sigma, (int, float)):
            jitter_sigma = torch.tensor(jitter_sigma)
        while jitter_sigma.dim() < signal_cumsum.dim():
            jitter_sigma = jitter_sigma.unsqueeze(-1)
        jitter_sigma = jitter_sigma.expand(signal_cumsum.shape)
        # jitter_sigma = jitter_sigma.unsqueeze(-1).unsqueeze(-1)
        
        jitter = torch.randn_like(signal_cumsum) * jitter_sigma
        jitter_running_mean = torch.nn.functional.avg_pool1d(jitter, mean_kernel_size, stride=1, padding=mean_kernel_size // 2)
        jitter -= jitter_running_mean

        # Step 3: Time with jitter
        time_delta = torch.ones_like(signal_cumsum)
        time_delta[..., :1] = 0
        time_delta_with_jitter = time_delta + jitter
        time_with_jitter = torch.cumsum(time_delta_with_jitter, dim=-1)
        time_with_jitter = torch.cummax(time_with_jitter, dim=-1)[0] # don't allow time to go back

        # # Step 4: Plot different jitters
        # for i in range(jitter.shape[0]):
        #     plt.plot(jitter[i, 0, :].detach().numpy(), alpha=0.5)
        # plt.show()

        # Step 5: Interpolation

        
        signal_cumsum_with_jitter = AbstractDataset.interpolate(signal_cumsum, time_with_jitter, time)
        signal_with_jitter = torch.diff(signal_cumsum_with_jitter, dim=-1) #, prepend=signal_cumsum_with_jitter[..., :1])

        return signal_with_jitter





class OrganicDataset(AbstractDataset):
    def __init__(self, 
                 df_path,
                 snr_distribution = np.inf,
                 total_duration_distribution = np.inf,
                 velocity_col='tangential_velocity',
                 trial_col='trial_n',
                 participant_col='participant',
                 seed=None,
                 quadratic_mean=None,
                 low_pass_filter=10,
                 small_values_threshold=None,
                 noise_mode='jitter',
                 purpose='test',
                 prepend_zeros=15,
                 absolute_velocity=False,
                 ):
        super().__init__()

        SR = 60
        FILTER_ORDER = 4

        if seed is None:
            seed = random.randint(0, 2**10)
        self.seed = seed # is it ever used?
        self.quadratic_mean = quadratic_mean
        self.low_pass_filter = low_pass_filter
        self.small_values_threshold = small_values_threshold
        self.noise_mode = noise_mode
        
        if low_pass_filter != np.inf:
            self.b, self.a = butter(FILTER_ORDER, low_pass_filter / SR * 2, btype='low', analog=False)
        else:
            self.b, self.a = None, None


        torch.manual_seed(self.seed)
        random.seed(self.seed)

        self.total_duration_distribution = self.convert_distribution(total_duration_distribution)
        self.snr_distribution = self.convert_distribution(snr_distribution)

        df = pd.read_csv(df_path)
        df = df[df[trial_col] >= 0] # no negative trial numbers
        participants = df[participant_col].unique()
        if participants.dtype == 'object':
            participants = participants.astype(str)
        participants = sorted(participants)

        if purpose == 'test':
            participants = participants[1::2]
        elif purpose == 'train':
            participants = participants[::2]
        else:
            raise ValueError(f"purpose should be 'test' or 'train', got {purpose}")
        
        df = df[df[participant_col].isin(participants)]
        df = df.reset_index(drop=True)

        self.purpose = purpose
        self.participants = participants

        self.prepend_zeros = prepend_zeros

        trials = sorted(df[trial_col].unique())
        total_trials = len(trials)
        total_samples = len(df)
        samples_per_trial = total_samples // total_trials

        self.trial2velocity_signal = df.groupby(trial_col)[velocity_col].apply(lambda x: x.values).to_dict()
        self.trial2participant = df.groupby(trial_col)[participant_col].apply(lambda x: x.mode().values[0]).to_dict()
        self.trials = list(self.trial2velocity_signal.keys())
        del df

        for trial in self.trials:
            self.trial2velocity_signal[trial] = np.pad(self.trial2velocity_signal[trial], (self.prepend_zeros, 0), mode='constant')

        self.absolute_velocity = absolute_velocity


        # self.join_n_trials = 1
        # if samples_per_trial < 10000:
        #     new_trials = []
        #     new_trial2velocity_signal = {}
        #     self.join_n_trials = 20000 // samples_per_trial

        #     for trial_n in range(0, total_trials, self.join_n_trials):
        #         trials_to_join = trials[trial_n:trial_n+self.join_n_trials]
        #         new_trial2velocity_signal[trial_n] = np.concatenate([np.pad(self.trial2velocity_signal[t], (self.prepend_zeros, 0), mode='constant') 
        #                                                              for t in trials_to_join])
        #         new_trials.append(trial_n)
        #     self.trial2velocity_signal = new_trial2velocity_signal
        #     self.trials = new_trials

    def __len__(self):
        return len(self.trial2velocity_signal)

    def __getitem__(self, idx):
        random.seed(idx + self.seed * len(self)) # while incrementing the seed, we can get the same seed for different idx
        torch.manual_seed(idx + self.seed * len(self))

        trial = self.trials[idx]
        x_array = self.trial2velocity_signal[trial]

        if self.absolute_velocity:
            x_array = np.abs(x_array)

        if self.low_pass_filter != np.inf:
            padlen = 3 * (max(len(self.b), len(self.a)) - 1)
            padlen = min(padlen, len(x_array) - 2)
            x_array = filtfilt(self.b, self.a, x_array, padlen=padlen).copy()

        # pad 30 zeros from the left
        # x_array = np.pad(x_array, (self.prepend_zeros, 0), mode='constant')

        x_clean = torch.tensor(x_array, dtype=self.dtype, device=self.device)

        total_duration = self.total_duration_distribution.sample().item()
        total_duration = int(min(total_duration, len(x_clean)))

        snr = self.snr_distribution.sample().item()

        start = random.randint(0, len(x_clean) - total_duration)
        if self.quadratic_mean is not None:
            x_clean_squared = x_clean**2
            original_quadratic_mean = torch.sqrt(torch.mean(x_clean_squared))
            if self.small_values_threshold is not None:
                x_clean_squared = x_clean_squared[x_clean_squared >= self.small_values_threshold * original_quadratic_mean**2]
                original_quadratic_mean = torch.sqrt(torch.mean(x_clean_squared))

            x_clean = x_clean / original_quadratic_mean * self.quadratic_mean
        x_clean = x_clean[start:start+total_duration]
        x_clean = x_clean.unsqueeze(0)

        if self.noise_mode == 'jitter' or self.noise_mode == 'mixed':
            x_noisy_jitter = self.add_jitter_noise(x_clean, snr=snr)
        if self.noise_mode == 'gaussian' or self.noise_mode == 'mixed':
            noise_coefficients = x_clean.std(-1) * (10 ** (-snr / 20))  
            noise_coefficients = noise_coefficients
            gaussian_noise = torch.randn_like(x_clean) * noise_coefficients
            x_noisy_gaussian = x_clean + gaussian_noise
        
        if self.noise_mode == 'jitter':
            x_noisy = x_noisy
        elif self.noise_mode == 'gaussian':
            x_noisy = x_noisy_gaussian
        elif self.noise_mode == 'mixed':
            jitter_part = random.random()
            gaussian_part = 1 - jitter_part
            x_noisy = x_noisy_jitter * jitter_part + x_noisy_gaussian * gaussian_part

        return x_noisy, x_clean, None
    
    @staticmethod
    def collate_fn(batch, pad_value=0, return_lengths=False):
    # Get the maximum sequence length
        lengths = [len(seq) for seq in batch]
        max_len = max(lengths)

        # Pad each sequence to the max length with the pad_value (0 or NaN)
        padded_batch = []
        for seq in batch:
            padded_seq = torch.full((max_len,), pad_value)  # Create padding
            padded_seq[:len(seq)] = torch.tensor(seq)  # Copy the original sequence
            padded_batch.append(padded_seq)

        if return_lengths:
            return torch.stack(padded_batch), torch.tensor(lengths)

        return torch.stack(padded_batch)

class SyntheticDataset(AbstractDataset):
    def __init__(self, 
                 total_duration_distribution = (100, 300), 
                 snr_distribution = (5, 20),
                 duration_distribution = (5, 30),
                 amplitude_distribution = (-300, 300),
                 amplitude_by_duration_distribution = None,
                 refractory_distribution = (0.5, 1.5),
                 joint_distribution = None,
                 joint_x = 'curr_amplitude',
                 joint_y = ('curr_duration', 'curr_refractory'),
                 joint_next_x = 'next_amplitude',
                 reconstruction_model = None,
                 max_submovements=None, 
                 num_samples=1000, 
                 dtype=DTYPE,
                 device=DEVICE,
                 seed=None,
                 batch_size=10,
                 refractory_mode='percentages',
                 min_allowed_refractory=3,
                 standardize=False,
                 one_sign_chance=0,
                 hard_refractory_chance=0,
                 easy_refractory_chance=0,
                 noise_mode='gaussian',
                 absolute_velocity=False,
                 make_symmetric=False,
                 values_require_symmetry=('curr_amplitude' , 'curr_mean_velocity', 'next_amplitude' , 'next_mean_velocity'),
                 ):
        super().__init__(device=device, dtype=dtype)

        if seed is None:
            seed = random.randint(0, 2**10)
        self.seed = seed # is it ever used?
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        self.max_submovements = max_submovements
        self.num_samples = num_samples
        self.dtype = dtype
        self.device = device

        self.batch_size = batch_size

        self.total_duration_distribution = self.convert_distribution(total_duration_distribution)
        self.snr_distribution = self.convert_distribution(snr_distribution)

        self.joint_distribution = joint_distribution
        if isinstance(self.joint_distribution, str):
            df = pd.read_csv(self.joint_distribution)
            # print(df.head())
            self.joint_distribution = ConditionalDistribution(df, x_label=joint_x, y_labels=joint_y, next_x_label=joint_next_x, make_symmetric=make_symmetric, values_require_symmetry=values_require_symmetry)

        if isinstance(self.joint_distribution, Distribution):
            self.duration_distribution =\
            self.amplitude_distribution =\
            self.amplitude_by_duration_distribution =\
            self.refractory_distribution = None
            self.one_sign_chance = \
            self.hard_refractory_chance = \
            self.easy_refractory_chance = 0
        else:
            self.duration_distribution = self.convert_distribution(duration_distribution)
            self.amplitude_distribution = self.convert_distribution(amplitude_distribution)
            self.amplitude_by_duration_distribution = self.convert_distribution(amplitude_by_duration_distribution)
            self.refractory_distribution = self.convert_distribution(refractory_distribution)

            self.one_sign_chance = one_sign_chance
            self.hard_refractory_chance = hard_refractory_chance
            self.easy_refractory_chance = easy_refractory_chance

            self.joint_distribution = None

        if reconstruction_model is None:
            base_reconstructor_params_copy = BASE_RECONSTRUCTOR_PARAMS.copy()
            base_reconstructor_params_copy['device'] = device
            base_reconstructor_params_copy['dtype'] = dtype

            if self.duration_distribution is not None:
                base_reconstructor_params_copy['duration_range'][0] = min(base_reconstructor_params_copy['duration_range'][0], self.duration_distribution.low.item())
                base_reconstructor_params_copy['duration_range'][1] = max(base_reconstructor_params_copy['duration_range'][1], self.duration_distribution.high.item())

            reconstruction_model = BASE_RECONSTRUCTOR(**base_reconstructor_params_copy)
        self.reconstruction_model = reconstruction_model


        self.refractory_mode = refractory_mode
        self.min_allowed_refractory = min_allowed_refractory

        self.do_standardize = standardize
        self.noise_mode = noise_mode
        self.absolute_velocity = absolute_velocity
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx, total_duration=None, snr=None):
        random.seed(idx + self.seed * len(self)) # while incrementing the seed, we can get the same seed for different idx
        torch.manual_seed(idx + self.seed * len(self))

        if total_duration is None:
            total_duration = self.total_duration_distribution.sample().int().item()
        # snr = self.snr_distribution.sample().item()
        if snr is None:
            snr = self.snr_distribution.sample([self.batch_size])
        elif isinstance(snr, (int, float)):
            snr = torch.tensor(snr, dtype=self.dtype, device=self.device)
            snr = snr.repeat(self.batch_size)

        if self.joint_distribution is not None:
            mask, durations_mask, amplitudes_mask = self.generate_labels_from_joint_distribution(total_duration)
        else:
            mask, durations_mask, amplitudes_mask = self.generate_labels_from_parameters(total_duration)

        if random.random() < self.one_sign_chance:
            amplitudes_mask = torch.abs(amplitudes_mask)
            sign = random.choice([-1, 1])
            amplitudes_mask = amplitudes_mask * sign

        if self.absolute_velocity:
            amplitudes_mask = torch.abs(amplitudes_mask)

        y_stacked = torch.stack([mask, amplitudes_mask, durations_mask], dim=0)
        y_stacked = y_stacked.transpose(0, 1)
        x_clean, _ = self.reconstruction_model(y_stacked, snr=None)

        # if self.absolute_velocity:
        #     x_clean = torch.abs(x_clean)

        if self.noise_mode == 'jitter' or self.noise_mode == 'mixed':
            x_noisy_jitter = self.add_jitter_noise(x_clean, snr=snr)
        if self.noise_mode == 'gaussian' or self.noise_mode == 'mixed':
            noise_coefficients = x_clean.std(-1) * (10 ** (-snr.unsqueeze(-1) / 20))  
            noise_coefficients = noise_coefficients.unsqueeze(-1)
            gaussian_noise = torch.randn_like(x_clean) * noise_coefficients
            x_noisy_gaussian = x_clean + gaussian_noise
        if self.noise_mode == 'gaussian':
            x_noisy = x_noisy_gaussian
        elif self.noise_mode == 'jitter':
            x_noisy = x_noisy_jitter
        elif self.noise_mode == 'mixed':
            jitter_part = random.random()
            gaussian_part = 1 - jitter_part
            x_noisy = x_noisy_jitter * jitter_part + x_noisy_gaussian * gaussian_part

        if self.do_standardize:
            x_noisy, mean_noisy, std_noisy = self.standardize(x_noisy)
            x_clean, _, _ = self.standardize(x_clean, mean_noisy, std_noisy)
            y_stacked[:, 1:2, :], _, _ = self.standardize(y_stacked[:, 1:2, :], mean_noisy, std_noisy) # scales amplitudes accordingly

        if self.do_standardize:
            MAX_ALLOWED_AMPLITUDE = 100
            max_amplitudes = y_stacked[:, 1, :].abs().max(-1)[0]
            x_noisy[max_amplitudes > MAX_ALLOWED_AMPLITUDE] = 0
            x_clean[max_amplitudes > MAX_ALLOWED_AMPLITUDE] = 0
            y_stacked[max_amplitudes > MAX_ALLOWED_AMPLITUDE] = 0

        if self.absolute_velocity:
            x_noisy = torch.abs(x_noisy)
            x_clean = torch.abs(x_clean)

        return x_noisy, x_clean, y_stacked
    
    @staticmethod
    def standardize_old(x, mean=0, std=None):
        """
        population mean = 0 is forced by default
        """

        if mean is None:
            mean = torch.mean(x, dim=-1)
            mean = mean.unsqueeze(-1)
        if std is None:
            std = torch.sqrt(torch.mean((x - mean)**2, dim=-1))
            std = std.unsqueeze(-1)

        std = torch.where(std == 0, torch.ones_like(std), std)


        x_standardized = (x - mean) / std
        return x_standardized, mean, std
    
    @staticmethod
    def standardize(x, mean=0, std=None, eps=1e-6):
        """
        population mean = 0 is forced by default
        Args:
            x: Input tensor
            mean: Pre-computed mean or 0
            std: Pre-computed standard deviation or None
            eps: Small constant for numerical stability
        """
        if mean is None:
            mean = torch.mean(x, dim=-1, keepdim=True)
        # elif not mean.dim() == x.dim():
        #     mean = mean.unsqueeze(-1)
            
        if std is None:
            std = torch.sqrt(torch.mean((x - mean)**2, dim=-1, keepdim=True))
        # elif not std.dim() == x.dim():
        #     std = std.unsqueeze(-1)
        
        # Add eps to std to prevent division by very small numbers
        std = torch.maximum(std, torch.ones_like(std) * eps)
        
        x_standardized = (x - mean) / std
        return x_standardized, mean, std


    def generate_labels_from_joint_distribution(self, total_duration, seed=None):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        mean_refractory = 6
        
        max_submovements = self.max_submovements
        if max_submovements is None:
            max_submovements = total_duration // mean_refractory
            max_submovements = int(max_submovements) 

        amplitudes = self.joint_distribution.x_chained_sample(self.batch_size, max_submovements)
        durations, refractories = self.joint_distribution.sample_y(amplitudes)

        amplitudes = torch.tensor(amplitudes, dtype=self.dtype, device=self.device)
        durations = torch.tensor(durations, dtype=self.dtype, device=self.device)
        refractories = torch.tensor(refractories, dtype=self.dtype, device=self.device) # .round()

        first_delays = torch.randint(0, 10, (self.batch_size,), dtype=self.dtype, device=self.device)
        refractories = refractories.roll(1)
        refractories = torch.clamp(refractories, self.min_allowed_refractory, None)
        refractories[:, 0] = first_delays

        condition1 = amplitudes[:, 1:] * amplitudes[:, :-1] < 0
        diff = durations[:, :-1] - (durations[:, 1:] + refractories[:, 1:])
        condition2 = diff > 0
        condition = condition1 & condition2

        if condition.sum() > 0:
            # I don't remember why I did this
            # probably shifting the refractory periods of submovements that are too close and have different signs
            shifts = torch.randint(0, 10, (condition.sum().item(),), device=self.device)
            # if self.refractory_mode == 'percentages':
            #     shifts *= durations[:, 1:][condition] # shifting according their own duration, not the previous one
            shifts -= shifts.min()
            refractories[:,1:][condition] += diff[condition] + shifts

        start_times = torch.cumsum(refractories, dim=1)

        mask, durations_mask, amplitudes_mask = self.labels_to_mask(start_times, durations, amplitudes, total_duration)

        return mask, durations_mask, amplitudes_mask
        
    def generate_labels_from_parameters(self, total_duration, seed=None):
        if seed is not None:
            # so we can set seed here, but we don't need to, what about the other seed in __init__?
            random.seed(seed)
            torch.manual_seed(seed)

        hard_refractory = easy_refractory = False

        if self.hard_refractory_chance > 0 or self.easy_refractory_chance > 0:
            coin = random.random()

            if coin < self.hard_refractory_chance:
                hard_refractory = True
            elif coin < self.hard_refractory_chance + self.easy_refractory_chance:
                easy_refractory = True

        # snr_distribution = convert_distribution(snr_distribution)
        mean_refractory = self.refractory_distribution.mean.item()
        if self.refractory_mode == 'percentages':
            mean_refractory *= self.duration_distribution.mean.item()
        
        if hard_refractory:
            mean_refractory /= 2
        elif easy_refractory:
            mean_refractory *= 2

        max_submovements = self.max_submovements
        if max_submovements is None:
            max_submovements = total_duration // mean_refractory
            max_submovements = int(max_submovements) # device problem?

        # start_times = torch.cumsum(refractories, dim=0)
        durations = self.duration_distribution.sample([self.batch_size, max_submovements])
        if self.amplitude_by_duration_distribution is not None:
            amplitudes = self.amplitude_by_duration_distribution.sample([self.batch_size, max_submovements]) * durations
        else:
            amplitudes = self.amplitude_distribution.sample([self.batch_size, max_submovements])
        refractories = self.refractory_distribution.sample([self.batch_size, max_submovements])

        if hard_refractory:
            refractories = refractories / 2
        elif easy_refractory:
            refractories = refractories * 2


        if self.refractory_mode == 'percentages':
            refractories[:, 1:] *= durations[:, :-1]
            refractories[:, 0] *= durations[:, 0]
            refractories += self.min_allowed_refractory - 0.5
        refractories = torch.clamp(refractories, self.min_allowed_refractory - 0.5, None)
            # refractories[refractories < self.min_allowed_refractory] += self.min_allowed_refractory
            # refractories = torch.clamp(refractories, self.min_allowed_refractory, None)

        # signs = sign_distribution.sample([max_submovements])
        # amplitudes = signs * amplitudes

        condition1 = amplitudes[:, 1:] * amplitudes[:, :-1] < 0
        diff = durations[:, :-1] - (durations[:, 1:] + refractories[:, 1:])
        condition2 = diff > 0
        condition = condition1 & condition2

        if condition.sum() > 0:
            # I don't remember why I did this
            # probably shifting the refractory periods of submovements that are too close 
            shifts = self.refractory_distribution.sample(torch.Size([condition.sum().item()]))
            if self.refractory_mode == 'percentages':
                shifts *= durations[:, 1:][condition] # shifting according their own duration, not the previous one
            shifts -= shifts.min()
            refractories[:,1:][condition] += diff[condition] + shifts

        start_times = torch.cumsum(refractories, dim=1)

        mask, durations_mask, amplitudes_mask = self.labels_to_mask(start_times, durations, amplitudes, total_duration)

        return mask, durations_mask, amplitudes_mask
    
    def labels_to_mask(self, start_times, durations, amplitudes, total_duration):

        start_times = start_times.to(torch.int64)

        start_times = torch.clamp(start_times, 0, total_duration-1) # put all extensive submovements at the end

        # print(self.batch_size, total_duration)
        mask = torch.zeros((self.batch_size, total_duration), dtype=self.dtype, device=self.device)
        durations_mask = torch.zeros_like(mask)
        amplitudes_mask = torch.zeros_like(mask)
        mask.scatter_(1, start_times, 1)
        durations_mask.scatter_(1, start_times, durations)
        amplitudes_mask.scatter_(1, start_times, amplitudes)

        # remove extensive submovements
        mask[:, -1] = 0
        durations_mask[:, -1] = 0
        amplitudes_mask[:, -1] = 0

        return mask, durations_mask, amplitudes_mask
    
class CombinedSyntheticDataset(AbstractDataset):
    def __init__(self, datasets, proportions=None, batch_size=10, total_duration_distribution=(100, 300), dtype=DTYPE, device=DEVICE, seed=None):
        assert len(datasets) > 0, "At least one dataset is required"

        
        if seed is not None:
            self.seed = seed
            # so we can set seed here, but we don't need to, what about the other seed in __init__?
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        # len_datasets = [len(dataset) for dataset in datasets]
        # assert len(set(len_datasets)) == 1, "All datasets must have the same length"

        # for dataset in datasets:
        #     if not isinstance(dataset, SyntheticDataset):
        #         raise ValueError("All datasets must be instances of SyntheticDataset")
            
        assert len(datasets) == len(proportions), "Number of datasets and proportions must match"

        len_datasets = [len(dataset) for dataset in datasets]
        assert len(set(len_datasets)) == 1, "All datasets must have the same length"

        self._len = len(datasets[0])
        self.dtype = dtype
        self.device = device

        self.datasets = datasets
        if proportions is None:
            proportions = [1] * len(datasets)
        proportions = np.array(proportions, dtype=np.float32)
        proportions = proportions / proportions.sum()
        self.proportions = proportions

        self.batch_size = batch_size
        self.batch_size_per_dataset = [int(batch_size * p) for p in proportions[:-1]]
        self.batch_size_per_dataset.append(batch_size - sum(self.batch_size_per_dataset))
        for batch_size, dataset in zip(self.batch_size_per_dataset, datasets):
            dataset.batch_size = batch_size

        self.total_duration_distribution =  self.convert_distribution(total_duration_distribution)
        
    def __len__(self):
        return self._len
    
    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, value):
        self._seed = value
        for dataset in self.datasets:
            dataset.seed = value

    def __getitem__(self, idx, total_duration=None, snr=None):
        random.seed(idx + self.seed * len(self)) # while incrementing the seed, we can get the same seed for different idx
        torch.manual_seed(idx + self.seed * len(self))
        np.random.seed(idx + self.seed * len(self))

        if total_duration is None:
            total_duration = self.total_duration_distribution.sample().int().item()

        x_noisy_parts = []
        x_clean_parts = []
        y_parts = []

        for dataset in self.datasets:
            batch_part = dataset.__getitem__(idx, total_duration, snr)
            x_noisy_parts.append(batch_part[0])
            x_clean_parts.append(batch_part[1])
            y_parts.append(batch_part[2])
        
        x_noisy = torch.cat(x_noisy_parts, dim=0)
        x_clean = torch.cat(x_clean_parts, dim=0)
        y = torch.cat(y_parts, dim=0)

        return x_noisy, x_clean, y
