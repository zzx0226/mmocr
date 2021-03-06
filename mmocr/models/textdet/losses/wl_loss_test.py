# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core import multi_apply
from torch import nn
import ptwt
import pywt
from mmocr.models.builder import LOSSES


@LOSSES.register_module()
class WLLoss_test(nn.Module):
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
        self.wavelet = pywt.Wavelet(wavelet_type)
        if self.wavelet_type == 'sym5' or self.wavelet_type == 'bior4.4' or self.wavelet_type == 'db5':
            self.num_cA = 20
            self.num_cD = [20, 31, 54]
        elif self.wavelet_type == 'bior3.1':
            self.num_cA = 15
            self.num_cD = [15, 27, 51]
        elif self.wavelet_type == 'bior3.5':
            self.num_cA = 22
            self.num_cD = [22, 33, 55]
        elif self.wavelet_type == 'coif3' or self.wavelet_type == 'rbio2.8':
            self.num_cA = 27
            self.num_cD = [27, 37, 58]
        elif self.wavelet_type == 'dmey':
            self.num_cA = 65
            self.num_cD = [65, 70, 80]

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
        loss_x = torch.tensor(0., device=device).float()
        loss_y = torch.tensor(0., device=device).float()

        for idx, loss in enumerate(losses):
            if idx == 0:
                loss_tr += sum(loss)
            elif idx == 1:
                loss_tcl += sum(loss)
            elif idx == 2:
                loss_x += sum(loss)
            else:
                loss_y += sum(loss)
        results = dict(loss_text=loss_tr, loss_center=loss_tcl, loss_x=loss_x, loss_y=loss_y)

        return results

    def forward_single(self, pred, gt):
        cls_pred = pred[0].permute(0, 2, 3, 1).contiguous()
        reg_pred = pred[1].permute(0, 2, 3, 1).contiguous()
        gt = gt.permute(0, 2, 3, 1).contiguous()

        k = self.num_cA
        tr_pred = cls_pred[:, :, :, :2].view(-1, 2)
        tcl_pred = cls_pred[:, :, :, 2:].view(-1, 2)
        wl_pred_real = reg_pred[:, :, :, 0:k].view(-1, k)
        wl_pred_imag = reg_pred[:, :, :, k + 1:-1].view(-1, k)
        x_center_pred = reg_pred[:, :, :, self.num_cA]
        y_center_pred = reg_pred[:, :, :, self.num_cA * 2 + 1]

        tr_mask = gt[:, :, :, :1].view(-1)
        tcl_mask = gt[:, :, :, 1:2].view(-1)
        train_mask = gt[:, :, :, 2:3].view(-1)
        real_map = gt[:, :, :, 3:3 + k].view(-1, k)
        imag_map = gt[:, :, :, 4 + k:-1].view(-1, k)
        x_center_gt = gt[:, :, :, 3 + self.num_cA]
        y_center_gt = gt[:, :, :, 3 + self.num_cA * 2 + 1]

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
        loss_x = torch.tensor(0.).float().to(device)
        loss_y = torch.tensor(0.).float().to(device)
        # loss_contour = torch.tensor(0.).float().to(device)
        if tr_train_mask.sum().item() > 0:
            # weight = (tr_mask[tr_train_mask.bool()].float() + tcl_mask[tr_train_mask.bool()].float()) / 2
            # weight = weight.contiguous().view(-1, 1)

            real_preds = wl_pred_real[tr_train_mask.bool()]
            real_gts = real_map[tr_train_mask.bool()]
            imag_preds = wl_pred_imag[tr_train_mask.bool()]
            imag_gts = imag_map[tr_train_mask.bool()]
            x_center_gts = x_center_gt.view(-1)[tr_train_mask.bool()]
            y_center_gts = y_center_gt.view(-1)[tr_train_mask.bool()]
            x_center_preds = x_center_pred.view(-1)[tr_train_mask.bool()]
            y_center_preds = y_center_pred.view(-1)[tr_train_mask.bool()]

            TotalNum = real_preds.shape[0]
            Matrix = torch.vstack((real_preds, real_gts, imag_preds, imag_gts))
            level1_coeff = torch.zeros(TotalNum * 4, self.num_cD[0]).float().cuda()
            level2_coeff = torch.zeros(TotalNum * 4, self.num_cD[1]).float().cuda()
            level3_coeff = torch.zeros(TotalNum * 4, self.num_cD[2]).float().cuda()
            wl_coeff = [Matrix, level1_coeff, level2_coeff, level3_coeff]
            rec = ptwt.waverec(wl_coeff, self.wavelet)

            pred_x = rec[:TotalNum] + x_center_preds.tile(100).reshape(100, -1).T
            gt_x = rec[TotalNum:TotalNum * 2] + x_center_gts.tile(100).reshape(100, -1).T
            pred_y = rec[TotalNum * 2:TotalNum * 3] + y_center_preds.tile(100).reshape(100, -1).T
            gt_y = rec[TotalNum * 3:] + y_center_gts.tile(100).reshape(100, -1).T

            # weight = torch.tile(torch.tensor(0.5), (1, TotalNum)).contiguous().view(-1, 1).cuda()
            # loss_x = torch.mean(weight * F.smooth_l1_loss(pred_x, gt_x, reduction='none'))  # / scale
            # loss_y = torch.mean(weight * F.smooth_l1_loss(pred_y, gt_y, reduction='none'))  # / scale

            # loss_contour = torch.mean(weight * F.smooth_l1_loss(pred_vectors, gt_vectors, reduction='none'))
            loss_x = torch.mean(F.smooth_l1_loss(pred_x, gt_x, reduction='none'))  # / scale
            loss_y = torch.mean(F.smooth_l1_loss(pred_y, gt_y, reduction='none'))  # / scale

        return loss_tr, loss_tcl, loss_x, loss_y

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
