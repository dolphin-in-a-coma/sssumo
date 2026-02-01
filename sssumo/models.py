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
    def __init__(self, duration_range=(4, 30), freeze_primitive_parameters=True, primitive_beta_mean=[0.5,0.0], primitive_beta_precision=[6.,0.0], device='cpu', dtype=torch.float32,
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

class ContinuousPrimitive(Primitive):
    def __init__(self, duration_range=(4, 30), freeze_parameters=True, beta_mean=(0.5, 0.00), beta_precision=(6., 0.0), device='cpu', dtype=torch.float32):
        # DONE: if sharpness can be order, it can, but I'll work with precision and mean.
        super(ContinuousPrimitive, self).__init__()
        device = torch.device(device)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.device = device
        self.dtype = dtype

        beta_mean = torch.tensor(beta_mean, device=self.device, dtype=self.dtype)
        beta_precision = torch.tensor(beta_precision, device=self.device, dtype=self.dtype)
        self.beta_mean = nn.Parameter(beta_mean, requires_grad=not freeze_parameters)
        self.beta_precision = nn.Parameter(beta_precision, requires_grad=not freeze_parameters)
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

        primitives = self.beta(duration)
        primitives *= auc

        return primitives
    
    def beta(self, duration):

        beta_means = self.beta_mean[0] + self.beta_mean[1] * duration
        beta_precisions = self.beta_precision[0] + self.beta_precision[1] * duration
        beta_alpha = beta_means * beta_precisions
        beta_beta = (1 - beta_means) * beta_precisions

        t = torch.arange(0.5, self.duration_range[1], 1, device=self.device, dtype=self.dtype)
        # t = torch.arange(0, self.duration_range[1], 1, device=self.device, dtype=self.dtype)
        # we suppose that a submovement starts and can end between two samples,
        # which should be average starting point

        t = t.reshape(*[1] * (len(duration.shape) - 1), -1)
        # DONE: probably need to fix dimensions
        normalized_t = torch.where(
            duration > 0,
            t / duration,
            torch.zeros_like(t) # Do I need gradient over t higher than duration actually?
        )

        normalized_t = torch.clamp(normalized_t, 0, 1)

        beta_bells = normalized_t ** (beta_alpha - 1) * (1 - normalized_t) ** (beta_beta - 1)
        debug = False
        if debug:
            for i in range(beta_bells.shape[-2]):
                if beta_bells[0, i, :].sum() != 0 and beta_bells[0, i, :].sum() < float('inf'):
                    plt.plot(beta_bells[0, i, :].detach().numpy(), label = f"Duration: {duration[0, i, 0].int().item()}, Sum: {beta_bells[0, i, :].sum().item()}")
                    print(beta_bells[0, i, :])
                    print(i)
            plt.legend()
            plt.show()

        beta_bells = torch.where(
            beta_bells.sum(dim=-1, keepdim=True) != 0,
            beta_bells / beta_bells.sum(dim=-1, keepdim=True),
            torch.zeros_like(beta_bells)
        )

        return beta_bells
    
    def beta_old(self, duration):

        beta_means = self.beta_mean[0] + self.beta_mean[1] * duration
        beta_precisions = self.beta_precision[0] + self.beta_precision[1] * duration
        beta_alpha = beta_means * beta_precisions
        beta_beta = (1 - beta_means) * beta_precisions

        t = torch.arange(0.5, self.duration_range[1], 1, dtype=torch.float32)
        # we suppose that a submovement starts and can end between two samples,
        # which should be average starting point

        t = t.reshape(*[1] * (len(duration.shape) - 1), -1)
        # DONE: probably need to fix dimensions

        beta_bells = (t / (duration + 1e-5)) ** (beta_alpha - 1) * (1 - t / (duration + 1e-5)) ** (beta_beta - 1)
        after_duration = t >= duration
        beta_bells[after_duration] = 0

        debug = False
        if debug:
            for i in range(beta_bells.shape[-2]):
                if beta_bells[0, i, :].sum() > 0 and beta_bells[0, i, :].sum() < float('inf'):
                    plt.plot(beta_bells[0, i, :].detach().numpy(), label = f"Duration: {duration[0, i, 0].int().item()}, Sum: {beta_bells[0, i, :].sum().item()}")
                    print(beta_bells[0, i, :])
                    print(i)
            plt.legend()
            plt.show()

        
        beta_bells = beta_bells / beta_bells.sum(dim=-1, keepdim=True)
        beta_bells = torch.nan_to_num(beta_bells, nan=0, posinf=0, neginf=0)
        beta_bells = torch.clamp(beta_bells, 0, None)

        return beta_bells
    

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

