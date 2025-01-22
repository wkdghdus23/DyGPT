import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.nn import Module, MSELoss, L1Loss
from torch import Tensor

def set_seed(seed: int) -> None:
    """
    Sets the random seed for reproducibility across different libraries.

    Args:
        seed (int): Random seed value to ensure reproducibility.

    Returns:
        None
    """
    # Sets the seed for Python's built-in random number generator
    random.seed(seed)
    # Sets the seed for NumPy's random number generator
    np.random.seed(seed)
    # Sets the seed for PyTorch's random number generator for CPU operations
    torch.manual_seed(seed)

    # Sets the seed for CUDA operations on GPU to ensure reproducibility across multiple GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # This setting optimizes CUDA kernel selection for performance, which may reduce reproducibility across different runs
    torch.backends.cudnn.benchmark = True

def adjust_learning_rate(optimizer: Optimizer, min_lr: float = 1e-6) -> None:
    """
    Adjusts the learning rate of an optimizer to ensure it does not fall below a specified minimum value.

    Args:
        optimizer (Optimizer): The optimizer whose learning rate needs adjustment.
        min_lr (float): The minimum allowable learning rate. Default is 1e-6.

    Returns:
        None
    """
    # Iterate through each parameter group in the optimizer
    for param_group in optimizer.param_groups:
        # Ensure the learning rate does not go below the specified minimum
        if param_group['lr'] < min_lr:
            param_group['lr'] = min_lr


def combined_loss(outputs: Tensor, labels: Tensor, alpha: float, 
                  loss_fn_mse: nn.Module = MSELoss(), 
                  loss_fn_mae: nn.Module = L1Loss()) -> Tensor:
    """
    Computes a combined loss that is a weighted sum of MSE and MAE losses.

    Args:
        outputs (Tensor): The model predictions.
        labels (Tensor): The true labels.
        alpha (float): The weight for combining the two loss functions. Should be in the range [0, 1].
        loss_fn_mse (nn.Module, optional): The Mean Squared Error (MSE) loss function. Default is nn.MSELoss().
        loss_fn_mae (nn.Module, optional): The Mean Absolute Error (MAE) loss function. Default is nn.L1Loss().

    Returns:
        Tensor: The combined loss value.
    """
    # Calculate the weighted combined loss
    mse_loss = loss_fn_mse(outputs, labels)
    mae_loss = loss_fn_mae(outputs, labels)
    combined = alpha * mse_loss + (1 - alpha) * mae_loss

    return combined


def loss_function(epoch: int, total_epochs: int) -> Module:
    """
    Selects an appropriate loss function based on the current epoch.

    Args:
        epoch (int): The current epoch number.
        total_epochs (int): The total number of training epochs.

    Returns:
        Module: The loss function for the current epoch.
    """
    if epoch + 1 <= int(total_epochs / 4):
        # Use Mean Squared Error (MSE) loss in the initial quarter of training
        loss_fn = nn.MSELoss()
    elif int(total_epochs / 4) < epoch + 1 <= int(total_epochs * 3 / 4):
        # Use a combined MSE and MAE loss in the middle half of training
        alpha = 1.0 - ((epoch - int(total_epochs / 4) + 1) / (total_epochs / 2))
        loss_fn = lambda outputs, labels: combined_loss(outputs, labels, alpha)
    else:
        # Use Mean Absolute Error (MAE) loss in the final quarter of training
        loss_fn = nn.L1Loss()

    return loss_fn
