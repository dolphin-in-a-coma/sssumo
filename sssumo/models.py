# %%
# Imports
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np

from abc import ABC, abstractmethod

import pdb

from torchviz import make_dot, make_dot_from_trace
# %%
# Models


class Detector(nn.Module, ABC):
    """Abstract class for detectors."""
    def __init__(self):
        super(Detector, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

class Reconstructor(nn.Module, ABC):
    """Abstract class for reconstructors."""
    def __init__(self):
        super(Reconstructor, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

class ProbabilityBinarizer(torch.autograd.Function):
    """Abstract class for binarizers.

    Parent class for:
    - STEBinarizer
    - AnnealedSigmoidBinarizer
    - GumbelBinarizer
    """

    def __init__(self):
        super(ProbabilityBinarizer, self).__init__()

    @staticmethod
    def forward(ctx, x):

        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass

class Primitive(nn.Module, ABC):
    """Abstract class for primitives.

    Parent class for:
    - ContinuousDurationPrimitive (Beta shape)
    - IntegerDurationPrimitive (Arbitrary)
    """

    def __init__(self):
        super(Primitive, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

# %%

class TDNNDetector(Detector):
    def __init__(self, dropout_rate=0.2, dilations=(1, 1, 1, 1, 1), kernel_sizes=(7, 7, 7, 7, 3), num_layers=5, channels=(1, 16, 32, 64, 128, 3), batchnorm=True):
        super(TDNNDetector, self).__init__()

        if isinstance(dilations, int):
            dilations = [1] + [dilations] * (num_layers - 2) + [1]
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_layers

        assert len(dilations) == num_layers, "dilations must have the same length as num_layers"
        assert len(kernel_sizes) == num_layers, "kernel_sizes must have the same length as num_layers"

        self.batchnorm = batchnorm

        self.conv_layers = nn.ModuleList()
        self.batchnorm_layers = nn.ModuleList()

        for i in range(num_layers):
            self.conv_layers.append(
                nn.Conv1d(in_channels=channels[i],
                          out_channels=channels[i+1],
                          kernel_size=kernel_sizes[i],
                          dilation=dilations[i],
                          padding=(kernel_sizes[i] - 1)*dilations[i]//2
                          )
            )

            
            if batchnorm and i < num_layers - 1:
                self.batchnorm_layers.append(nn.BatchNorm1d(channels[i+1]))

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if i < len(self.conv_layers) - 1:
                x = self.relu(x)
                x = self.dropout(x)

                if self.batchnorm:
                    x = self.batchnorm_layers[i](x)

        x[:, 0, :] = self.sigmoid(x[:, 0, :]) # onset probability vector
        if x.shape[1] == 4:
            x[:, 3, :] = self.sigmoid(x[:, 3, :]) # mask vector
        return x
    
class TDNNDetectorOld(Detector):
    def __init__(self, dropout_rate=0.2):
        super(TDNNDetector, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, dilation=1, padding=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, dilation=1, padding=3)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, dilation=1, padding=3)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, dilation=1, padding=3)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=3, kernel_size=3, dilation=1, padding=1)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.batchnorm4 = nn.BatchNorm1d(128)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.batchnorm1(self.dropout(x))
        
        x = self.relu(self.conv2(x))
        x = self.batchnorm2(self.dropout(x))

        x = self.relu(self.conv3(x))
        x = self.batchnorm3(self.dropout(x))

        x = self.relu(self.conv4(x))
        x = self.batchnorm4(self.dropout(x))

        x = self.conv5(x)
        x[:, 0, :] = self.sigmoid(x[:, 0, :])
        return x

class STEContinuousReconstructor(Reconstructor):
    def __init__(self, duration_range=(4, 30), freeze_primitive_parameters=True,
                 primitive_beta_mean=[0.5, 0.0], primitive_beta_precision=[6., 0.0],
                 primitive_family='beta',
                 primitive_gaussian_centre=[0.5, 0.0], primitive_gaussian_half_width=[2.5, 0.0],
                 primitive_lgnb_mu=[0.0, 0.0], primitive_lgnb_sigma=[0.8, 0.0],
                 device='cpu', dtype=torch.float32,
                 gradient_for_detection=False):
        super(STEContinuousReconstructor, self).__init__()
        device = torch.device(device)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
            # I don't know what it is, but works!
        self.device = device
        self.dtype = dtype

        self.binarizer = STEBinarizer
        self.primitive = ContinuousPrimitive(
            duration_range=duration_range,
            freeze_parameters=freeze_primitive_parameters,
            beta_mean=primitive_beta_mean,
            beta_precision=primitive_beta_precision,
            family=primitive_family,
            gaussian_centre=primitive_gaussian_centre,
            gaussian_half_width=primitive_gaussian_half_width,
            lgnb_mu=primitive_lgnb_mu,
            lgnb_sigma=primitive_lgnb_sigma,
            device=device,
            dtype=dtype
        ).to(device, dtype)

        self.gradient_for_detection = gradient_for_detection

    def forward(self, x, snr=None, only_peaks=True):
        if x.device != self.device or x.dtype != self.dtype:
            x = x.to(self.device, self.dtype)
        mask = x[:, 0, :]
        if x.shape[1] == 4:
            reconstruction_mask = x[:, 3, :]
        else:
            reconstruction_mask = None

        if self.gradient_for_detection == 'NegativeOnly':
            binarized_mask = self.binarizer.apply(mask, True, only_peaks)
        else:
            binarized_mask = self.binarizer.apply(mask, False, only_peaks)
        
        if not self.gradient_for_detection:
            binarized_mask = binarized_mask.detach()
        # elif self.gradient_for_detection == 'NegativeOnly':

        if reconstruction_mask is not None:
            binarized_reconstruction_mask = self.binarizer.apply(reconstruction_mask, False, only_peaks)

        auc = x[:, 1, :]
        duration = x[:, 2, :]

        auc_duration = torch.stack([auc, duration], dim=1)

        primitives = self.primitive(auc_duration)
        primitives *= binarized_mask.unsqueeze(-1)

        if reconstruction_mask is not None:
            primitives *= binarized_reconstruction_mask.unsqueeze(-1)



        for i in range(primitives.shape[-1]):
            primitives[:, :, i] = torch.roll(primitives[:, :, i], i, dims=-1)
            primitives[:, :i, i] = 0

        reconstructed_signal = primitives.sum(dim=-1)

        noisy_reconstructed_signal = reconstructed_signal.detach().clone()

        if snr is not None:
            noise_coefficients = reconstructed_signal.std(-1) * (10 ** (-snr / 20))  
            noise_coefficients = noise_coefficients.unsqueeze(-1)
            noise = torch.randn_like(reconstructed_signal) * noise_coefficients
            noisy_reconstructed_signal = reconstructed_signal + noise

        reconstructed_signal = reconstructed_signal.unsqueeze(1)
        noisy_reconstructed_signal = noisy_reconstructed_signal.unsqueeze(1)

        return reconstructed_signal, noisy_reconstructed_signal
    
class BetaShape(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta_mean, beta_precision):
        beta_alpha = beta_mean * beta_precision
        beta_beta = (1 - beta_mean) * beta_precision

        beta = x ** (beta_alpha - 1) * (1 - x) ** (beta_beta - 1)
        ctx.save_for_backward(x, beta_alpha, beta_beta)
        return beta
    
    @staticmethod
    def backward(ctx, grad_output):
        x, beta_alpha, beta_beta = ctx.saved_tensors
        grad_x = grad_output * beta_alpha * (x ** (beta_alpha - 1)) * ((1 - x) ** (beta_beta - 1) - beta_beta * x ** (beta_alpha) * (1 - x) ** (beta_beta - 2))
        return grad_x, None, None
    
class STEBinarizer(torch.autograd.Function):
    def __init__(self):
        super(STEBinarizer, self).__init__()

    @staticmethod
    def forward(ctx, x, only_negative_backprop=False, only_peaks=True):
        """
        Forward pass of the STEBinarizer.
        Binarizes the input `x` with a custom condition and saves context.

        Args:
            ctx: Context object to store information for the backward pass.
            x (torch.Tensor): Input tensor.
            only_negative_backprop (bool): If True, only allow negative gradients.

        Returns:
            torch.Tensor: Binarized tensor.
        """
        # Clone the input to avoid modifying the original tensor
        input_left = x[:, :-1]
        input_right = x[:, 1:]
        x = x.clone()

        # Apply binarization logic
        if only_peaks:
            x[:, 1:][input_right <= input_left] = 0 
            # x[:, 1:][(input_right <= input_left) & (input_right != 1)] = 0 # made ones allowed to be one after another
            x[:, :-1][input_left < input_right] = 0 
            # if two guys have the same value, the right one set to zero

        # Save tensors and flags to context for backward
        ctx.save_for_backward(x)
        ctx.only_negative_backprop = only_negative_backprop
        ctx.only_peaks = only_peaks

        return (x >= 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for STEBinarizer.
        Passes gradients with optional modifications based on flags.

        Args:
            ctx: Context object storing forward pass information.
            grad_output (torch.Tensor): Gradient of the loss w.r.t. the output.

        Returns:
            torch.Tensor: Gradient of the loss w.r.t. the input.
        """
        # Retrieve saved tensors and flags
        x, = ctx.saved_tensors

        # Modify gradients if only_negative_backprop is True
        if ctx.only_negative_backprop:
            grad_output = torch.clamp(grad_output, max=0)  # Keep only negative gradients
        
        if ctx.only_peaks:
            grad_output[x <= 0] = 0  # Zero gradients for inputs that were zeroed

        # Return gradients for x; no gradient for only_negative_backprop (non-tensor)
        return grad_output, None, None

PRIMITIVE_FAMILIES = ('beta', 'gaussian', 'lgnb')

# Which parameter pair each family actually reads. Only the active family's pair is
# ever made trainable: the others get no gradient, so leaving them in the optimiser
# would only clutter the logs with parameters that cannot move.
FAMILY_PARAMETERS = {
    'beta': ('beta_mean', 'beta_precision'),
    'gaussian': ('gaussian_centre', 'gaussian_half_width'),
    'lgnb': ('lgnb_mu', 'lgnb_sigma'),
}


class ContinuousPrimitive(Primitive):
    """A unit-area velocity pulse of finite support, on a fixed time grid.

    Every family is a probability density on normalised time ``s = t / duration``
    restricted to ``s in (0, 1)``, discretised on the half-sample grid
    ``t = 0.5, 1.5, ...`` and renormalised to sum to one. The pulse is then scaled
    by the area (``auc``), so the amplitude channel means the same thing whichever
    family is in use -- that invariant is what makes families comparable, and any
    new family must preserve it.

    Families (``family=``):

    ``beta``
        ``s**(a - 1) * (1 - s)**(b - 1)`` with ``a = mean * precision`` and
        ``b = (1 - mean) * precision``. ``mean=0.5, precision=6`` gives Beta(3, 3),
        which *is* the minimum-jerk profile ``30 s^2 (1-s)^2`` -- the two agree to
        ~2e-8 on this grid, so minimum jerk is a frozen special case of this family
        rather than a separate implementation.
    ``gaussian``
        A Gaussian truncated to the support, ``exp(-0.5 z^2)`` with
        ``z = (s - centre) * 2 * half_width``, so the support spans
        ``+/- half_width`` standard deviations. Symmetric, but with tails and a
        peak sharpness that Beta(3, 3) cannot match.
    ``lgnb``
        The support-bounded lognormal: a logit-normal density,
        ``exp(-0.5 ((logit(s) - mu) / sigma)^2) / (s (1 - s))``. Asymmetric for
        ``mu != 0``, and finite-support by construction.

    Each family parameter is a pair ``(intercept, slope)`` evaluated as
    ``intercept + slope * duration``, so a shape may vary with pulse duration. The
    slope is 0 in every shipped config.

    Two numerical points that are load-bearing:

    * Support is imposed by an explicit ``t < duration`` mask, *not* by letting
      ``(1 - s)**(b - 1)`` vanish at ``s = 1``. The old code relied on the latter,
      which silently restricts the usable parameter range: ``b < 1`` makes that
      term diverge and produces inf pulses and NaN gradients. With the mask, the
      whole ``(mean, precision)`` plane is usable.
    * ``s`` is clamped to ``[eps, 1 - eps]`` before exponentiation so that
      ``d/da s**(a-1) = s**(a-1) log s`` stays finite. With ``eps=1e-6`` the clamp
      never binds on an in-support sample (the extreme grid points are
      ``0.5/duration`` and ``1 - 0.5/duration``), so this is a no-op for the
      forward pass and only regularises the gradient.
    """

    def __init__(self, duration_range=(4, 30), freeze_parameters=True,
                 beta_mean=(0.5, 0.00), beta_precision=(6., 0.0),
                 family='beta',
                 gaussian_centre=(0.5, 0.0), gaussian_half_width=(2.5, 0.0),
                 lgnb_mu=(0.0, 0.0), lgnb_sigma=(0.8, 0.0),
                 eps=1e-6,
                 device='cpu', dtype=torch.float32):
        super(ContinuousPrimitive, self).__init__()
        device = torch.device(device)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.device = device
        self.dtype = dtype

        family = str(family).lower()
        if family not in PRIMITIVE_FAMILIES:
            raise ValueError(f"unknown primitive family {family!r}, "
                             f"expected one of {PRIMITIVE_FAMILIES}")
        self.family = family
        self.eps = eps

        active = FAMILY_PARAMETERS[self.family]

        def _pair(name, value):
            tensor = torch.as_tensor(value, device=self.device, dtype=self.dtype)
            if tensor.dim() == 0:
                tensor = torch.stack([tensor, torch.zeros_like(tensor)])
            trainable = (not freeze_parameters) and name in active
            return nn.Parameter(tensor, requires_grad=trainable)

        # Kept as attributes for every family so that checkpoints and configs stay
        # loadable across families; only the active family's parameters are read.
        self.beta_mean = _pair('beta_mean', beta_mean)
        self.beta_precision = _pair('beta_precision', beta_precision)
        self.gaussian_centre = _pair('gaussian_centre', gaussian_centre)
        self.gaussian_half_width = _pair('gaussian_half_width', gaussian_half_width)
        self.lgnb_mu = _pair('lgnb_mu', lgnb_mu)
        self.lgnb_sigma = _pair('lgnb_sigma', lgnb_sigma)

        self.duration_range = duration_range

    def forward(self, x):
        if x.device != self.device or x.dtype != self.dtype:
            x = x.to(self.device, self.dtype)
        auc = x[:, 0, :]
        duration = x[:, 1, :]
        duration = torch.where(
            duration == 0,
            duration,
            torch.clamp(duration, min=self.duration_range[0], max=self.duration_range[1])
        )

        auc = auc.unsqueeze(-1)
        duration = duration.unsqueeze(-1)

        primitives = self.profile(duration)
        primitives *= auc

        return primitives

    def _grid(self, duration):
        """Half-sample grid, normalised time, and the in-support mask."""
        t = torch.arange(0.5, self.duration_range[1], 1, device=self.device, dtype=self.dtype)
        # a submovement starts and can end between two samples, so the half-sample
        # offset is the average starting point
        t = t.reshape(*[1] * (len(duration.shape) - 1), -1)

        inside = (t < duration) & (duration > 0)

        normalized_t = torch.where(
            duration > 0,
            t / torch.clamp(duration, min=1e-6),
            torch.zeros_like(t)
        )
        # clamped only to keep gradients finite at the ends; see the class docstring
        normalized_t = torch.clamp(normalized_t, self.eps, 1 - self.eps)

        return normalized_t, inside

    def _linear(self, pair, duration):
        return pair[0] + pair[1] * duration

    def profile(self, duration):
        """Unit-sum pulses of the active family, one per (batch, time) entry."""
        normalized_t, inside = self._grid(duration)

        if self.family == 'beta':
            mean = self._linear(self.beta_mean, duration)
            precision = self._linear(self.beta_precision, duration)
            alpha = mean * precision
            beta = (1 - mean) * precision
            bells = normalized_t ** (alpha - 1) * (1 - normalized_t) ** (beta - 1)

        elif self.family == 'gaussian':
            centre = self._linear(self.gaussian_centre, duration)
            half_width = self._linear(self.gaussian_half_width, duration)
            # s spans one unit across the support, so scaling by 2 * half_width
            # puts the support edges at +/- half_width standard deviations
            z = (normalized_t - centre) * 2 * half_width
            bells = torch.exp(-0.5 * z ** 2)

        elif self.family == 'lgnb':
            mu = self._linear(self.lgnb_mu, duration)
            sigma = self._linear(self.lgnb_sigma, duration)
            logit = torch.log(normalized_t) - torch.log1p(-normalized_t)
            bells = (torch.exp(-0.5 * ((logit - mu) / sigma) ** 2)
                     / (normalized_t * (1 - normalized_t)))

        else:  # unreachable: guarded in __init__
            raise ValueError(self.family)

        bells = torch.where(inside, bells, torch.zeros_like(bells))

        total = bells.sum(dim=-1, keepdim=True)
        bells = torch.where(total != 0, bells / total, torch.zeros_like(bells))

        return bells

    def beta(self, duration):
        """Deprecated alias for :meth:`profile`, kept for notebooks."""
        return self.profile(duration)


# deprecated

def beta_function_special(assymetry, excentricity, t_scaled):
    # alpha, beta, x
    # DONE: implement torch thing, as a function or a class? inhereted?
    return None

class LinearBetaPrimitives(nn.Module):
    def __init__(self, duration_range=(4, 30)):
        super(LinearBetaPrimitives, self).__init__()
        self.assymetry_intercept = torch.tensor(1) # check if tensor should be put somewhere or grad True should be specified
        self.assymetry_slope = torch.tensor(1)
        self.excentricity_intercept = torch.tensor(1)
        self.excentricity_slope = torch.tensor(1)

        self.duration_range = torch.arange(duration_range)

        self.scaled_t = torch.zeros(len(self.duration_range), self.duration_range[-1]) # N x t

        # not so optimal, but who cares?
        for i, duration in enumerate(self.duration_range):
            scaled_time_for_step = 1/(duration+1)
            self.scaled_t[:duration] = torch.linspace(scaled_time_for_step/2, 1-scaled_time_for_step/2, duration)
            # can be shifted to start at step 1/2 and end at n-1/2, I guess it's made shifted now
            # CHECK: need to check again

    def forward(self):
        # CHECK: here x is not needed?
        assymetries = self.assymetry_intercept + self.assymetry_slope * self.duration_range
        excentricities = self.excentricity_intercept + self.excentricity_slope * self.duration_range

        primitives = beta_function_special(assymetries, excentricities, self.scaled_t)

        return primitives

