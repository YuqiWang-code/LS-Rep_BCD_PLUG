"""
A2Net_LWGANet - Binary Change Detection Model

Model: A2Net + LWGANet-L0 backbone (+ optional AFD feature distillation)
"""

from .a2net import A2Net_LWGANet_L0, A2Net_LWGANet_L2
from .losses.combined_loss import build_loss

__all__ = ['A2Net_LWGANet_L0', 'A2Net_LWGANet_L2', 'build_loss']
