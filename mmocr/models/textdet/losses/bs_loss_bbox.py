# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core import multi_apply, reduce_mean
from torch import nn
from mmocr.models.builder import LOSSES
from mmdet.models.builder import build_loss


@LOSSES.register_module()
class BSLoss_bbox(nn.Module):

    def __init__(self, bs_degree, cp_num, ohem_ratio=3.):
        super().__init__()
        self.bs_degree = bs_degree
        self.cp_num = cp_num
        self.ohem_ratio = ohem_ratio
        loss_bbox = dict(type='CIoULoss', loss_weight=1.0)
        # loss_bbox = dict(type='IoULoss', loss_weight=1.0)
        self.loss_bbox = build_loss(loss_bbox)

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
        loss_bbox = torch.tensor(0., device=device).float()

        for idx, loss in enumerate(losses):
            if idx == 0:
                loss_tr += sum(loss)
            elif idx == 1:
                loss_tcl += sum(loss)
            elif idx == 2:
                loss_reg_x += sum(loss)
            elif idx == 3:
                loss_reg_y += sum(loss)
            else:
                loss_bbox += sum(loss)

        results = dict(loss_text=loss_tr,
                       loss_center=loss_tcl,
                       loss_reg_x=loss_reg_x,
                       loss_reg_y=loss_reg_y,
                       loss_bbox=loss_bbox)

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

        # bboxes = gt[:, :, :, 3 + k * 2:].view(-1, 4)

        device = x_map.device
        tr_train_mask = train_mask * tr_mask

        # tr loss
        loss_tr = self.ohem(tr_pred, tr_mask.long(), train_mask.long())

        # tcl loss
        loss_tcl = torch.tensor(0.).float().to(device)
        loss_reg_x = torch.tensor(0.).float().to(device)
        loss_reg_y = torch.tensor(0.).float().to(device)
        loss_bbox = torch.tensor(0.).float().to(device)

        tr_neg_mask = 1 - tr_train_mask
        if tr_train_mask.sum().item() > 0:
            loss_tcl_pos = F.cross_entropy(tcl_pred[tr_train_mask.bool()], tcl_mask[tr_train_mask.bool()].long())
            loss_tcl_neg = F.cross_entropy(tcl_pred[tr_neg_mask.bool()], tcl_mask[tr_neg_mask.bool()].long())
            loss_tcl = loss_tcl_pos + 0.5 * loss_tcl_neg

        # regression loss
        if tr_train_mask.sum().item() > 0:
            t_pred = torch.min(y_pred[tr_train_mask.bool()][:, :5], dim=1).values
            b_pred = torch.max(y_pred[tr_train_mask.bool()][:, 5:], dim=1).values
            r_pred = torch.max(x_pred[tr_train_mask.bool()], dim=1).values
            l_pred = torch.min(x_pred[tr_train_mask.bool()], dim=1).values
            # bbox_pred = torch.cat((t_pred, l_pred, b_pred, r_pred), dim=0).reshape(-1, 4)
            bbox_pred = torch.cat((l_pred, t_pred, r_pred, b_pred), dim=0).reshape(-1, 4)

            t_gt = torch.min(y_map[tr_train_mask.bool()][:, :5], dim=1).values
            b_gt = torch.max(y_map[tr_train_mask.bool()][:, 5:], dim=1).values
            r_gt = torch.max(x_map[tr_train_mask.bool()], dim=1).values
            l_gt = torch.min(x_map[tr_train_mask.bool()], dim=1).values
            # bbox_gt = torch.cat((t_gt, l_gt, b_gt, r_gt), dim=0).reshape(-1, 4)
            bbox_gt = torch.cat((l_gt, t_gt, r_gt, b_gt), dim=0).reshape(-1, 4)
            # centerness_targets = self.centerness_target(bbox_gt)
            # centerness weighted iou loss
            # centerness_denorm = max(reduce_mean(centerness_targets.sum().detach()), 1e-6)
            # loss_bbox = self.loss_bbox(bbox_pred, bbox_gt, weight=centerness_targets, avg_factor=centerness_denorm)
            loss_bbox = self.loss_bbox(bbox_pred, bbox_gt)
            weight = (tr_mask[tr_train_mask.bool()].float() + tcl_mask[tr_train_mask.bool()].float()) / 2  # 10
            weight = weight.contiguous().view(-1, 1)

            loss_reg_x = torch.mean(
                weight *
                F.smooth_l1_loss(x_map[tr_train_mask.bool()], x_pred[tr_train_mask.bool()], reduction='none'))  # / scale
            loss_reg_y = torch.mean(
                weight *
                F.smooth_l1_loss(y_map[tr_train_mask.bool()], y_pred[tr_train_mask.bool()], reduction='none'))  # / scale

        return loss_tr, loss_tcl, loss_reg_x, loss_reg_y, loss_bbox

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

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (top_bottom.min(dim=-1)[0] /
                                                                                            top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)