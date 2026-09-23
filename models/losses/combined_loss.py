"""
BCE + Dice combined loss for binary change detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy + Dice Loss
    """
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets)
        inter = (inputs * targets).sum()
        eps = 1e-5
        dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
        return bce + 1 - dice


def build_loss():
    """
    Build loss function for A2Net training
    Returns BCEDiceLoss
    """
    return BCEDiceLoss()
