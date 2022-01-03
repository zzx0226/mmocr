# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core import multi_apply
from torch import nn
import pywt
from mmocr.models.builder import LOSSES


@LOSSES.register_module()
class WLLoss(nn.Module):
    """The class for implementing FCENet loss.

    FCENet(CVPR2021): `Fourier Contour Embedding for Arbitrary-shaped Text
    Detection <https://arxiv.org/abs/2104.10442>`_

    Args:
        fourier_degree (int) : The maximum Fourier transform degree k.
        num_sample (int) : The sampling points number of regression
            loss. If it is too small, fcenet tends to be overfitting.
        ohem_ratio (float): the negative/positive ratio in OHEM.
    """
    def __init__(self, wavelet_type='haar', ohem_ratio=3.):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.ohem_ratio = ohem_ratio

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
        loss_wl = torch.tensor(0., device=device).float()

        for idx, loss in enumerate(losses):
            if idx == 0:
                loss_tr += sum(loss)
            elif idx == 1:
                loss_tcl += sum(loss)
            else:
                loss_wl += sum(loss)
        results = dict(loss_text=loss_tr, loss_center=loss_tcl, loss_wl=loss_wl)

        return results

    def forward_single(self, pred, gt):
        cls_pred = pred[0].permute(0, 2, 3, 1).contiguous()
        reg_pred = pred[1].permute(0, 2, 3, 1).contiguous()
        gt = gt.permute(0, 2, 3, 1).contiguous()

        # if gt.shape[1] == 100:
        #     scale = 8
        # elif gt.shape[1] == 50:
        #     scale = 16
        # elif gt.shape[1] == 25:
        #     scale = 32

        tr_pred = cls_pred[:, :, :, :2].view(-1, 2)
        tcl_pred = cls_pred[:, :, :, 2:].view(-1, 2)
        wl_coeffs_pred = reg_pred[:, :, :, :].view(-1, 28)

        tr_mask = gt[:, :, :, :1].view(-1)
        tcl_mask = gt[:, :, :, 1:2].view(-1)
        train_mask = gt[:, :, :, 2:3].view(-1)
        wl_coeffs_target = gt[:, :, :, 3:].view(-1, 28)

        tr_train_mask = train_mask * tr_mask
        device = wl_coeffs_target.device
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
        loss_wl = torch.tensor(0.).float().to(device)
        if tr_train_mask.sum().item() > 0:
            weight = (tr_mask[tr_train_mask.bool()].float() + tcl_mask[tr_train_mask.bool()].float()) / 5
            weight = weight.contiguous().view(-1, 1)

            # points_pred=self.Coeffs2Poly()

            loss_wl = torch.mean(weight * F.smooth_l1_loss(
                wl_coeffs_pred[tr_train_mask.bool()], wl_coeffs_target[tr_train_mask.bool()], reduction='none'))

        return loss_tr, loss_tcl, loss_wl

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
