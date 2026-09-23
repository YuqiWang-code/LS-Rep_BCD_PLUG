"""
AFD (Attention-based Feature Distillation) Module
Source: LSNet (IEEE TIP 2023)
Pay Attention to Features, Transfer Learn Faster CNNs
https://openreview.net/pdf?id=ryxyCeHtPB
"""

import torch
import torch.nn as nn


class AFD_semantic(nn.Module):
    '''
    Channel Attention AFD
    '''

    def __init__(self, in_channels, att_f):
        super(AFD_semantic, self).__init__()
        mid_channels = int(in_channels * att_f)

        self.attention = nn.Sequential(*[
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 3, 1, 1, bias=True)
        ])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t, eps=1e-6):
        fm_t_pooled = self.avg_pool(fm_t)
        rho = self.attention(fm_t_pooled)
        rho = torch.sigmoid(rho.squeeze())
        rho = rho / torch.sum(rho, dim=1, keepdim=True)

        fm_s_norm = torch.norm(fm_s, dim=(2, 3), keepdim=True)
        fm_s = torch.div(fm_s, fm_s_norm + eps)
        fm_t_norm = torch.norm(fm_t, dim=(2, 3), keepdim=True)
        fm_t = torch.div(fm_t, fm_t_norm + eps)

        loss = rho * torch.pow(fm_s - fm_t, 2).mean(dim=(2, 3))
        loss = loss.sum(1).mean(0)

        return loss


class AFD_spatial(nn.Module):
    '''
    Spatial Attention AFD
    '''

    def __init__(self, in_channels):
        super(AFD_spatial, self).__init__()

        self.attention = nn.Sequential(*[
            nn.Conv2d(in_channels, 1, 3, 1, 1)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t, eps=1e-6):
        rho = self.attention(fm_t)
        rho = torch.sigmoid(rho)
        rho = rho / torch.sum(rho, dim=(2, 3), keepdim=True)

        fm_s_norm = torch.norm(fm_s, dim=1, keepdim=True)
        fm_s = torch.div(fm_s, fm_s_norm + eps)
        fm_t_norm = torch.norm(fm_t, dim=1, keepdim=True)
        fm_t = torch.div(fm_t, fm_t_norm + eps)

        loss = rho * torch.pow(fm_s - fm_t, 2).mean(dim=1, keepdim=True)
        loss = torch.sum(loss, dim=(2, 3)).mean(0)

        return loss
