# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core import multi_apply
from torch import nn

from mmocr.models.builder import LOSSES
from geomdl import BSpline
from geomdl import knotvector
# from chamfer_distance import ChamferDistance
from .chamfer_distance import ChamferDistance, chamfer_distance
# chamfer_dist = ChamferDistance()

# from mmocr.models.builder import build_loss


@LOSSES.register_module()
class BSLoss_tb_new(nn.Module):
    """The class for implementing FCENet loss.

    FCENet(CVPR2021): `Fourier Contour Embedding for Arbitrary-shaped Text
    Detection <https://arxiv.org/abs/2104.10442>`_

    Args:
        fourier_degree (int) : The maximum Fourier transform degree k.
        num_sample (int) : The sampling points number of regression
            loss. If it is too small, fcenet tends to be overfitting.
        ohem_ratio (float): the negative/positive ratio in OHEM.
    """

    def __init__(self, bs_degree, cp_num, ohem_ratio=3.):
        super().__init__()
        self.bs_degree = bs_degree
        self.cp_num = cp_num
        self.ohem_ratio = ohem_ratio
        # loss_bbox = dict(type='IoULoss', loss_weight=1.0)
        # self.loss_bbox = build_loss(loss_bbox)

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
        loss_reg_x = torch.tensor(0., device=device).float()
        loss_reg_y = torch.tensor(0., device=device).float()
        # loss_top = torch.tensor(0., device=device).float()
        # loss_bottom = torch.tensor(0., device=device).float()

        for idx, loss in enumerate(losses):
            if idx == 0:
                loss_tr += sum(loss)
            elif idx == 1:
                loss_tcl += sum(loss)
            elif idx == 2:
                loss_reg_x += sum(loss)
            elif idx == 3:
                loss_reg_y += sum(loss)
            # elif idx == 4:
            #     loss_top += sum(loss)
            # else:
            #     loss_bottom += sum(loss)

        results = dict(
            loss_text=loss_tr,
            loss_center=loss_tcl,
            loss_reg_x=loss_reg_x,
            loss_reg_y=loss_reg_y,
            # loss_top=loss_top / 100,
            # loss_bottom=loss_bottom / 100,
        )

        return results

    def forward_single(self, pred, gt):
        cls_pred = pred[0].permute(0, 2, 3, 1).contiguous()
        reg_pred = pred[1].permute(0, 2, 3, 1).contiguous()

        gt = gt.permute(0, 2, 3, 1).contiguous()

        k = self.cp_num * 2
        tr_pred = cls_pred[:, :, :, :2].view(-1, 2)
        tcl_pred = cls_pred[:, :, :, 2:].view(-1, 2)
        x_pred = reg_pred[:, :, :, 0:k].view(-1, k)
        y_pred = reg_pred[:, :, :, k:].view(-1, k)

        tr_mask = gt[:, :, :, :1].view(-1)
        tcl_mask = gt[:, :, :, 1:2].view(-1)
        train_mask = gt[:, :, :, 2:3].view(-1)
        x_map = gt[:, :, :, 3:3 + k].view(-1, k)
        y_map = gt[:, :, :, 3 + k:3 + k * 2].view(-1, k)
        device = x_map.device

        # x_map_temp = torch.cat((x_map.reshape(-1, int(k / 2), 2), torch.zeros((x_map.shape[0], int(k / 2), 1)).cuda()),
        #                        dim=2).to(device)
        # y_map_temp = torch.cat((y_map.reshape(-1, int(k / 2), 2), torch.zeros((y_map.shape[0], int(k / 2), 1)).cuda()),
        #                        dim=2).to(device)

        # x_pred_temp = torch.cat((x_pred.reshape(-1, int(k / 2), 2), torch.zeros((x_pred.shape[0], int(k / 2), 1)).cuda()),
        #                         dim=2).to(device)
        # y_pred_temp = torch.cat((y_pred.reshape(-1, int(k / 2), 2), torch.zeros((y_pred.shape[0], int(k / 2), 1)).cuda()),
        #                         dim=2).to(device)

        tr_train_mask = train_mask * tr_mask

        # tr loss
        loss_tr = self.ohem(tr_pred, tr_mask.long(), train_mask.long())

        # tcl loss
        loss_tcl = torch.tensor(0.).float().to(device)
        loss_reg_x = torch.tensor(0.).float().to(device)
        loss_reg_y = torch.tensor(0.).float().to(device)
        # top_loss = torch.tensor(0.).float().to(device)
        # bottom_loss = torch.tensor(0.).float().to(device)

        tr_neg_mask = 1 - tr_train_mask
        if tr_train_mask.sum().item() > 0:
            loss_tcl_pos = F.cross_entropy(tcl_pred[tr_train_mask.bool()], tcl_mask[tr_train_mask.bool()].long())
            loss_tcl_neg = F.cross_entropy(tcl_pred[tr_neg_mask.bool()], tcl_mask[tr_neg_mask.bool()].long())
            loss_tcl = loss_tcl_pos + 0.5 * loss_tcl_neg

        # regression loss
        if tr_train_mask.sum().item() > 0:
            weight = (tr_mask[tr_train_mask.bool()].float() + tcl_mask[tr_train_mask.bool()].float()) / 2  # 10
            weight = weight.contiguous().view(-1, 1)

            loss_reg_x = torch.mean(
                weight *
                F.smooth_l1_loss(x_map[tr_train_mask.bool()], x_pred[tr_train_mask.bool()], reduction='none'))  # / scale
            loss_reg_y = torch.mean(
                weight *
                F.smooth_l1_loss(y_map[tr_train_mask.bool()], y_pred[tr_train_mask.bool()], reduction='none'))  # / scale
            #chamfer_distance chamfer_dist
            # dist1, dist2, idx1, idx2 = chamfer_distance(x_map_temp.float(), x_pred_temp.float())
            # top_loss = (torch.mean(dist1)) + (torch.mean(dist2))

            # dist1, dist2, idx1, idx2 = chamfer_distance(y_map_temp.float(), y_pred_temp.float())
            # bottom_loss = (torch.mean(dist1)) + (torch.mean(dist2))
        return loss_tr, loss_tcl, loss_reg_x, loss_reg_y  #, top_loss, bottom_loss

    def cal_contour(self, ContrlPoint):
        if (ContrlPoint == 0).all():
            return np.zeros(40)
        ContrlPoint = ContrlPoint.reshape(-1, 2)
        crv = BSpline.Curve()
        crv.degree = self.bs_degree
        crv.ctrlpts = ContrlPoint.tolist()
        crv.knotvector = knotvector.generate(crv.degree, crv.ctrlpts_size)
        crv.sample_size = 20
        points = np.array(crv.evalpts).flatten()
        return points

    #     pass
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
