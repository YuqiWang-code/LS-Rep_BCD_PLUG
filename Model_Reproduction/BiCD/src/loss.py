import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from scipy.spatial import distance

from collections import OrderedDict
from src.dataset.dataset import get_transforms
from src.dataset import transforms as T
import cv2
import os
import numpy as np
import math
from src.model.binary_version_HFA_blank_decoder.Backbone_ours import RPReLU


def aux_criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


class criterion_CEloss(nn.Module):
    def __init__(self, weight=None):
        super(criterion_CEloss, self).__init__()
        self.loss = nn.NLLLoss(weight)

    def forward(self, output, target):
        return self.loss(F.log_softmax(output, dim=1), target)


class Bicriterion(nn.Module):
    def __init__(self, mask_loss):
        super(Bicriterion, self).__init__()
        self.mask_loss = mask_loss

    def forward(self, output, target):
        mask_loss = self.mask_loss(output['out'], target)
        return mask_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.ep = 1e-8

    def forward(self, output, target):
        pred = (F.softmax(output, dim=1)[:, 0]).float()
        target = (target > 0).float()
        intersection = 2 * torch.sum(pred * target) + self.ep
        union = torch.sum(pred) + torch.sum(target) + self.ep
        loss = 1 - intersection / union
        return loss


class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        if isinstance(pred, OrderedDict):
            pred = pred['out']
        if true.dim() == 3:
            true.unsqueeze_(1)
            true = torch.cat([1 - true, true], dim=1)
        true = true.float()
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BCEDiceLoss(nn.Module):
    def __init__(self, loss_weight):
        super(BCEDiceLoss, self).__init__()
        self.eps = 1e-5
        self.celoss = nn.CrossEntropyLoss(loss_weight)

    def forward(self, inputs, targets):
        bce = self.celoss(inputs, targets)
        inter = (inputs * targets).sum()
        dice = (2 * inter + self.eps) / (inputs.sum() + targets.sum() + self.eps)
        return bce + 1 - dice


class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets)
        return bce


class regularization_loss(nn.Module):
    def __init__(self):
        super(regularization_loss, self).__init__()
        self.REGULARIZATION_LOSS_WEIGHT = 1.

    def forward(self, output):
        loss = 0.
        output_list = [output[i] for i in range(output.shape[0])]
        f_num = len(output_list)
        for f_m in output_list:
            loss += (f_m ** 2).mean()

        loss *= self.REGULARIZATION_LOSS_WEIGHT
        loss /= float(f_num)
        return loss


class HFALoss_drtanet_behind(nn.Module):
    def __init__(self):
        super(HFALoss_drtanet_behind, self).__init__()
        self.loss = nn.L1Loss()
        self.smooothloss = nn.SmoothL1Loss(reduction='mean', beta=0.5)
        self.loss2 = nn.MSELoss()
        self.revert = T.RevertNormalize_loss(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def forward(self, img, rebuild1, rebuild2, mask_gt):
        img_t0, img_t1 = torch.split(img, 3, 1)
        img_t0 = img_t0.detach()
        img_t1 = img_t1.detach()
        mask_gt1 = mask_gt.unsqueeze(1).detach()
        rebuild = torch.abs(self.revert(rebuild2) - self.revert(rebuild1))
        change_origin = torch.abs(self.revert(img_t1) - self.revert(img_t0)).detach()

        zero_tensor = torch.zeros_like(rebuild).detach()
        eps = 1e-8
        nochange0 = self.loss(rebuild1, img_t0)
        nochange1 = self.loss(rebuild2, img_t1)
        nochange2 = self.loss(rebuild * (1 - mask_gt1), zero_tensor)
        change = self.loss(rebuild * mask_gt1, change_origin * mask_gt1)

        out = nochange0 + nochange1 + nochange2 + change
        return out


class HFALoss_ours_behind(nn.Module):
    def __init__(self):
        super(HFALoss_ours_behind, self).__init__()
        self.loss = nn.L1Loss()
        self.smooothloss = nn.SmoothL1Loss(reduction='mean', beta=0.5)
        self.loss2 = nn.MSELoss()

        self.normalize = T.Normalize_loss(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.revert = T.RevertNormalize_loss(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        self.epoch = 0
        self.min_weight = 0.1
        self.T_max = 75

    def cosine_annealing_weight(self):
        return self.min_weight + (1 - self.min_weight) / 2 * (1 + math.cos(max((math.pi * self.epoch), 0) / self.T_max))

    def forward(self, img, rebuild0, rebuild1, mask_gt, epoch):
        if epoch is not None:
            self.epoch = epoch

        weight = 1.5 * self.cosine_annealing_weight()

        num_f = len(img)
        tmp_img = T.UnbindImages()(img)
        tmp_img_list = list(tmp_img)
        tmp_rebuild0 = T.UnbindImages()(rebuild0)
        tmp_rebuild1 = T.UnbindImages()(rebuild1)
        tmp_rebuild0_list = list(tmp_rebuild0)
        tmp_rebuild1_list = list(tmp_rebuild1)

        mask_gt1 = mask_gt.unsqueeze(1)
        tmp_mask_gt1 = T.UnbindImages()(mask_gt1)
        mask_gt1_list = list(tmp_mask_gt1)

        eps = 0.00001
        zero_tensor = torch.zeros_like(tmp_rebuild0_list[0])
        out = 0.0
        for i in range(num_f):
            tmp_img_list[i] = tmp_img_list[i]

            img_t0, img_t1 = torch.split(tmp_img_list[i], 3, 0)

            tmp_rebuild = torch.abs(self.revert(tmp_rebuild1_list[i]) - self.revert(tmp_rebuild0_list[i]))

            rebuild = self.normalize(tmp_rebuild)
            change_origin = self.normalize(torch.abs(self.revert(img_t1) - self.revert(img_t0)))

            nochange0 = self.loss(tmp_rebuild0_list[i], img_t0.detach())
            nochange1 = self.loss(tmp_rebuild1_list[i], img_t1.detach())
            nochange2 = self.loss(rebuild * (1 - mask_gt1_list[i].detach()), self.normalize(zero_tensor.detach()))
            change = self.loss(rebuild * mask_gt1_list[i].detach(), (change_origin * mask_gt1_list[i]).detach())

            out += weight * (nochange0 + nochange1) + nochange2 + change

        out_final = out / num_f
        return out_final


class HFALoss_ours_behind_new(nn.Module):
    def __init__(self):
        super(HFALoss_ours_behind_new, self).__init__()
        self.loss = nn.L1Loss()
        self.smooothloss = nn.SmoothL1Loss(reduction='mean', beta=0.5)
        self.loss2 = nn.MSELoss()

        self.normalize = T.Normalize_loss(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.revert = T.RevertNormalize_loss(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        self.epoch = 0
        self.min_weight = 0.1
        self.T_max = 50

    def cosine_annealing_weight(self):
        return self.min_weight + (1 - self.min_weight) / 2 * (1 + math.cos(max((math.pi * self.epoch), 0) / self.T_max))

    def forward(self, img, rebuild0, rebuild1, rebuild_change, mask_gt, epoch):
        if epoch is not None:
            self.epoch = epoch

        weight =  self.cosine_annealing_weight()

        num_f = len(img)
        tmp_img = T.UnbindImages()(img)
        tmp_img_list = list(tmp_img)
        tmp_rebuild0 = T.UnbindImages()(rebuild0)
        tmp_rebuild1 = T.UnbindImages()(rebuild1)
        tmp_rebuild0_list = list(tmp_rebuild0)
        tmp_rebuild1_list = list(tmp_rebuild1)
        tmp_rebuild_change = T.UnbindImages()(rebuild_change)
        tmp_rebuild_change_list = list(tmp_rebuild_change)

        mask_gt1 = mask_gt.unsqueeze(1)
        tmp_mask_gt1 = T.UnbindImages()(mask_gt1)
        mask_gt1_list = list(tmp_mask_gt1)

        eps = 0.00001
        zero_tensor = torch.zeros_like(tmp_rebuild0_list[0])
        out = 0.0
        for i in range(num_f):
            tmp_img = tmp_img_list[i]

            img_t0, img_t1 = torch.split(tmp_img, 3, 0)

            rebuild = tmp_rebuild_change_list[i]

            change_origin = self.normalize(torch.max(self.revert(img_t0), self.revert(img_t1)) - torch.min(self.revert(img_t0), self.revert(img_t1)))

            nochange0 = self.loss(tmp_rebuild0_list[i], img_t0.detach())
            nochange1 = self.loss(tmp_rebuild1_list[i], img_t1.detach())
            nochange2 = self.loss(rebuild * (1 - mask_gt1_list[i].detach()), self.normalize(zero_tensor.detach()))
            change = self.loss(rebuild * mask_gt1_list[i].detach(), (change_origin * mask_gt1_list[i]).detach())

            out += weight * (nochange0 + nochange1) + (1-weight) * (nochange2 + change)

        out_final = out / num_f
        return out_final


def get_loss(name, loss_weight=None):
    print("Loss: {}".format(name))
    if name == 'focalloss':
        BCEseg = nn.BCEWithLogitsLoss()
        FLseg = FocalLoss(BCEseg)
        return FLseg
    elif name == 'ce':
        return nn.CrossEntropyLoss(loss_weight, ignore_index=255)
    elif name == 'aux':
        return aux_criterion
    elif name == 'bce':
        return BCE()
    elif name == 'bcedice':
        return BCEDiceLoss(loss_weight)
    elif name == 'bi':
        mask_loss = nn.CrossEntropyLoss(loss_weight)
        return Bicriterion(mask_loss)
    elif name == 'bi_dice':
        dice_loss = DiceLoss()
        return Bicriterion(dice_loss)
    elif name == 'HFALoss_drtanet_behind':
        return HFALoss_drtanet_behind()
    elif name == 'HFALoss_ours_behind':
        return HFALoss_ours_behind_new()
    elif name == 'regular_loss':
        return regularization_loss()
    else:
        raise ValueError(name)