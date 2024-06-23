import torch

def dice_loss(input, target, eps=1e-6):
    """
    Compute the Dice coefficient for two float tensors.

    Args:
        input (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Ground truth tensor.
        eps (float): A small constant to avoid division by zero.

    Returns:
        float: Dice coefficient.
    """
    # Flatten the input and target tensors
    input_flat = input.view(-1)
    target_flat = target.view(-1)

    # Calculate intersection and union
    intersection = torch.sum(input_flat * target_flat)
    union = torch.sum(input_flat) + torch.sum(target_flat)

    # Calculate Dice coefficient
    dice = (2. * intersection + eps) / (union + eps)
    
    return 1 - dice.item()