"""
A2Net - Change Detection Network

Core architecture combining:
- LWGANet backbone (shared dual-temporal)
- Neighbor Feature Aggregation (SWA)
- Temporal Fusion Module (TFM)
- Multi-scale Decoder with Supervised Attention
- Optional AFD (Attention-based Feature Distillation, training-only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.lwganet import LWGANet_L0_1242_e32_k11_GELU, LWGANet_L2_1242_e96_k11_RELU
from .backbone.afd import AFD_semantic, AFD_spatial
from .decoder.a2net_decoder import (
    NeighborFeatureAggregation,
    TemporalFusionModule,
    Decoder
)
from .plugins.graft_plug import GRAFTPlug


class _A2NetBase(nn.Module):
    """Shared A2Net logic: backbone + SWA + TFM + Decoder + optional AFD."""

    def __init__(self, backbone, channels, en_d=32, use_afd=False, graft_cfg=None):
        super().__init__()
        self.backbone = backbone
        self.mid_d = en_d * 2
        self.swa = NeighborFeatureAggregation(channels, self.mid_d)
        self.tfm = TemporalFusionModule(self.mid_d, self.mid_d)
        self.decoder = Decoder(self.mid_d)

        self.use_afd = use_afd
        if use_afd:
            # After SWA aggregation all features become mid_d = 64
            self.AFD_semantic_4 = AFD_semantic(self.mid_d, 0.0625)
            self.AFD_semantic_3 = AFD_semantic(self.mid_d, 0.0625)
            self.AFD_spatial_2 = AFD_spatial(self.mid_d)
            self.AFD_spatial_1 = AFD_spatial(self.mid_d)

        # GRAFT-PLUG：训练期外挂，switch_to_deploy() 整体删除
        self.use_graft = graft_cfg is not None
        if graft_cfg is not None:
            self.graft = GRAFTPlug(**graft_cfg)

    # ------------------------------------------------------------------
    # Sub-routines
    # ------------------------------------------------------------------

    def _compute_afd_loss(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):
        """Bidirectional AFD loss on SWA-aggregated (64-ch) features."""
        return (
            self.AFD_semantic_4(x1_5, x2_5.detach()) +
            self.AFD_semantic_4(x2_5, x1_5.detach()) +
            self.AFD_semantic_3(x1_4, x2_4.detach()) +
            self.AFD_semantic_3(x2_4, x1_4.detach()) +
            self.AFD_spatial_2(x1_3, x2_3.detach()) +
            self.AFD_spatial_2(x2_3, x1_3.detach()) +
            self.AFD_spatial_1(x1_2, x2_2.detach()) +
            self.AFD_spatial_1(x2_2, x1_2.detach())
        )

    def _forward_main_path(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):
        """Temporal fusion + FPN decoder + multi-scale change maps."""
        c2, c3, c4, c5 = self.tfm(x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5)
        p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5 = self.decoder(c2, c3, c4, c5)

        mask_p2 = torch.sigmoid(F.interpolate(mask_p2, scale_factor=(4, 4), mode='bilinear'))
        mask_p3 = torch.sigmoid(F.interpolate(mask_p3, scale_factor=(8, 8), mode='bilinear'))
        mask_p4 = torch.sigmoid(F.interpolate(mask_p4, scale_factor=(16, 16), mode='bilinear'))
        mask_p5 = torch.sigmoid(F.interpolate(mask_p5, scale_factor=(32, 32), mode='bilinear'))
        return mask_p2, mask_p3, mask_p4, mask_p5

    # ------------------------------------------------------------------
    # Forward / deploy
    # ------------------------------------------------------------------

    def forward(self, x1, x2, target=None):
        feats1 = tuple(self.backbone(x1))
        feats2 = tuple(self.backbone(x2))

        aux_losses = {}

        # GRAFT-PLUG：挂在 raw stage 之外，只产生辅助损失，绝不进主路
        if self.training and self.use_graft:
            aux_losses.update(self.graft(feats1, feats2, target))

        x1_2, x1_3, x1_4, x1_5 = feats1
        x2_2, x2_3, x2_4, x2_5 = feats2

        x1_2, x1_3, x1_4, x1_5 = self.swa(x1_2, x1_3, x1_4, x1_5)
        x2_2, x2_3, x2_4, x2_5 = self.swa(x2_2, x2_3, x2_4, x2_5)

        if self.training and self.use_afd:
            aux_losses["afd"] = self._compute_afd_loss(
                x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5
            )

        masks = self._forward_main_path(
            x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5
        )

        if self.training:
            return masks, aux_losses
        return masks

    def switch_to_deploy(self):
        """Idempotently remove AFD and GRAFT; main path remains intact."""
        if getattr(self, "use_afd", False):
            for name in ("AFD_semantic_4", "AFD_semantic_3",
                         "AFD_spatial_2", "AFD_spatial_1"):
                if hasattr(self, name):
                    delattr(self, name)
            self.use_afd = False
            print("AFD modules removed for deployment")
        if getattr(self, "use_graft", False):
            if hasattr(self, "graft"):
                del self.graft
            self.use_graft = False
            print("GRAFT modules removed for deployment")
        return self


class A2Net_LWGANet_L0(_A2NetBase):
    """
    A2Net with LWGANet-L0 backbone (lightweight version).

    Params: 2.91M (inference) / 2.92M (with AFD)
    FLOPs: 2.75G (inference)
    """
    def __init__(self, pretrained=True, use_afd=False, graft_cfg=None):
        backbone = LWGANet_L0_1242_e32_k11_GELU(pretrained=pretrained)
        channels = [32, 32, 64, 128, 256]
        super().__init__(backbone, channels, en_d=32, use_afd=use_afd, graft_cfg=graft_cfg)


class A2Net_LWGANet_L2(_A2NetBase):
    """
    A2Net with LWGANet-L2 backbone (larger version).
    """
    def __init__(self, pretrained=True, use_afd=False, graft_cfg=None):
        backbone = LWGANet_L2_1242_e96_k11_RELU(pretrained=pretrained)
        channels = [96, 96, 192, 384, 768]
        super().__init__(backbone, channels, en_d=32, use_afd=use_afd, graft_cfg=graft_cfg)
