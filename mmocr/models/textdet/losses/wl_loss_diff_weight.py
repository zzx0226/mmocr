# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core import multi_apply
from torch import nn
import pywt
from mmocr.models.builder import LOSSES


@LOSSES.register_module()
class WLLoss_diff_weight(nn.Module):
    """The class for implementing FCENet loss.

    FCENet(CVPR2021): `Fourier Contour Embedding for Arbitrary-shaped Text
    Detection <https://arxiv.org/abs/2104.10442>`_

    Args:
        fourier_degree (int) : The maximum Fourier transform degree k.
        num_sample (int) : The sampling points number of regression
            loss. If it is too small, fcenet tends to be overfitting.
        ohem_ratio (float): the negative/positive ratio in OHEM.
    """

    def __init__(self, wavelet_type='sym5', ohem_ratio=3.):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.ohem_ratio = ohem_ratio
        self.weight = torch.cat([
            torch.tensor([1] * 20),
            torch.tensor([0.5] * 20),
            torch.tensor([0.25] * 31),
            torch.tensor([0.125] * 54),
            torch.tensor([1])
        ],
                                dim=0).cuda()

    def forward(self, preds, _, p3_maps, p4_maps, p5_maps):
        """Compute FCENet loss.

        Args:
            preds (list[list[Tensor]]): The outer list indicates images
                in a batch, and the inner list indicates the classification
                prediction map (with shape :math:`(N, C, H, W)`) and
                regression map (with shape :math:`(N, C, H, W)`).
            p3_maps (list[ndarray]): List of leval 3 ground truth target map
                with shape :math:`(C, H, W)`.
            p4_maps (list[ndarray]): List of leval 4 ground truth target map
                with shape :math:`(C, H, W)`.
            p5_maps (list[ndarray]): List of leval 5 ground truth target map
                with shape :math:`(C, H, W)`.

        Returns:
            dict:  A loss dict with ``loss_text``, ``loss_center``,
            ``loss_reg_x`` and ``loss_reg_y``.
        """
        assert isinstance(preds, list)
        # assert p3_maps[0].shape[0] == 4 * self.fourier_degree + 5,\
        #     'fourier degree not equal in FCEhead and FCEtarget'

        device = preds[0][0].device
        # to tensor
        gts = [p3_maps, p4_maps, p5_maps]
        for idx, maps in enumerate(gts):
            gts[idx] = torch.from_numpy(np.stack(maps)).float().to(device)

        losses = multi_apply(self.forward_single, preds, gts)

        loss_tr = torch.tensor(0., device=device).float()
        loss_tcl = torch.tensor(0., device=device).float()
        loss_real = torch.tensor(0., device=device).float()
        loss_imag = torch.tensor(0., device=device).float()

        for idx, loss in enumerate(losses):
            if idx == 0:
                loss_tr += sum(loss)
            elif idx == 1:
                loss_tcl += sum(loss)
            elif idx == 2:
                loss_real += sum(loss)
            else:
                loss_imag += sum(loss)
        results = dict(loss_text=loss_tr, loss_center=loss_tcl, loss_real=loss_real, loss_imag=loss_imag)

        return results

    def forward_single(self, pred, gt):
        cls_pred = pred[0].permute(0, 2, 3, 1).contiguous()
        reg_pred = pred[1].permute(0, 2, 3, 1).contiguous()
        gt = gt.permute(0, 2, 3, 1).contiguous()

        k = 126
        tr_pred = cls_pred[:, :, :, :2].view(-1, 2)
        tcl_pred = cls_pred[:, :, :, 2:].view(-1, 2)
        wl_pred_real = reg_pred[:, :, :, :k].view(-1, k)
        wl_pred_imag = reg_pred[:, :, :, k:].view(-1, k)
        # wl_pred_imag = reg_pred[:, :, :, k + 1:-1].view(-1, k)
        # x_center_pred = reg_pred[:, :, :, 125]
        # y_center_pred = reg_pred[:, :, :, 251]

        tr_mask = gt[:, :, :, :1].view(-1)
        tcl_mask = gt[:, :, :, 1:2].view(-1)
        train_mask = gt[:, :, :, 2:3].view(-1)
        real_map = gt[:, :, :, 3:3 + k].view(-1, k)
        imag_map = gt[:, :, :, 3 + k:].view(-1, k)
        # imag_map = gt[:, :, :, 4 + k:-1].view(-1, k)
        # x_center_gt = gt[:, :, :, 3 + 125]
        # y_center_gt = gt[:, :, :, 3 + 251]

        tr_train_mask = train_mask * tr_mask
        device = real_map.device
        # tr loss
        loss_tr = self.ohem(tr_pred, tr_mask.long(), train_mask.long())

        # tcl loss
        loss_tcl = torch.tensor(0.).float().to(device)
        tr_neg_mask = 1 - tr_train_mask
        if tr_train_mask.sum().item() > 0:
            loss_tcl_pos = F.cross_entropy(tcl_pred[tr_train_mask.bool()], tcl_mask[tr_train_mask.bool()].long())
            loss_tcl_neg = F.cross_entropy(tcl_pred[tr_neg_mask.bool()], tcl_mask[tr_neg_mask.bool()].long())
            loss_tcl = loss_tcl_pos + 0.5 * loss_tcl_neg

        # regression loss
        loss_real = torch.tensor(0.).float().to(device)
        loss_imag = torch.tensor(0.).float().to(device)
        if tr_train_mask.sum().item() > 0:
            weight = torch.tile(self.weight, (len(wl_pred_real[tr_train_mask.bool()]), 1))
            # weight = weight.contiguous()

            loss_real = torch.mean(
                weight * F.smooth_l1_loss(wl_pred_real[tr_train_mask.bool()], real_map[tr_train_mask.bool()], reduction='none'))
            loss_imag = torch.mean(
                weight * F.smooth_l1_loss(wl_pred_imag[tr_train_mask.bool()], imag_map[tr_train_mask.bool()], reduction='none'))
            # loss_real = torch.mean(
            #     F.smooth_l1_loss(wl_pred_real[tr_train_mask.bool()], real_map[tr_train_mask.bool()], reduction='none'))
            # loss_imag = torch.mean(
            #     F.smooth_l1_loss(wl_pred_imag[tr_train_mask.bool()], imag_map[tr_train_mask.bool()], reduction='none'))
        return loss_tr, loss_tcl, loss_real, loss_imag

    def Coeffs2Poly(self, WL_Coeffs):
        wl_coeff = WL_Coeffs.reshape(-1, 7)
        cA, cH, cV, cD = wl_coeff[0], wl_coeff[1], wl_coeff[2], wl_coeff[3]
        points = pywt.waverec2((cA, (cH, cV, cD)), self.wavelet_type)
        return points

    def ohem(self, predict, target, train_mask):
        device = train_mask.device
        pos = (target * train_mask).bool()
        neg = ((1 - target) * train_mask).bool()

        n_pos = pos.float().sum()

        if n_pos.item() > 0:
            loss_pos = F.cross_entropy(predict[pos], target[pos], reduction='sum')
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = min(int(neg.float().sum().item()), int(self.ohem_ratio * n_pos.float()))
        else:
            loss_pos = torch.tensor(0.).to(device)
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = 100
        if len(loss_neg) > n_neg:
            loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()
