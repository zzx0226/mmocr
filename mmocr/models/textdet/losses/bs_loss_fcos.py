# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core import multi_apply, reduce_mean
from torch import nn
from mmocr.models.builder import LOSSES, build_loss

INF = 1e8


@LOSSES.register_module()
class BSLoss_fcos(nn.Module):
    """The class for implementing FCENet loss.

    FCENet(CVPR2021): `Fourier Contour Embedding for Arbitrary-shaped Text
    Detection <https://arxiv.org/abs/2104.10442>`_

    Args:
        fourier_degree (int) : The maximum Fourier transform degree k.
        num_sample (int) : The sampling points number of regression
            loss. If it is too small, fcenet tends to be overfitting.
        ohem_ratio (float): the negative/positive ratio in OHEM.
    """
    def __init__(self,
                 loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 bs_degree=4,
                 cp_num=14,
                 ohem_ratio=3.):
        super().__init__()
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)

        self.bs_degree = bs_degree
        self.cp_num = cp_num
        self.ohem_ratio = ohem_ratio

    def forward(self,
                cls_scores,
                bbox_preds,
                bs_preds,
                centernesses,
                gt_bboxes,
                gt_cp,
                gt_labels,
                img_metas,
                gt_bboxes_ignore=None):

        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(bs_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(featmap_sizes,
                                                            dtype=bbox_preds[0].dtype,
                                                            device=bbox_preds[0].device)
        labels, bbox_targets, bs_targets = self.get_targets(all_level_points, gt_bboxes, gt_cp, gt_labels)
        #, cp_target
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds]
        flatten_centerness_pred = [centerness.permute(0, 2, 3, 1).reshape(-1) for centerness in centernesses]

        flatten_bs_preds = [bs_pred.permute(0, 2, 3, 1).reshape(-1, self.cp_num[0] * 2) for bs_pred in bs_preds]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness_pred = torch.cat(flatten_centerness_pred)

        flatten_bs_preds = torch.cat(flatten_bs_preds)

        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)

        flatten_bs_targets = torch.cat(bs_targets)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bs_preds = flatten_bs_preds[pos_inds]

        pos_centerness_pred = flatten_centerness_pred[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]

        pos_bs_targets = flatten_bs_targets[pos_inds]

        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(pos_decoded_bbox_preds,
                                       pos_decoded_target_preds,
                                       weight=pos_centerness_targets,
                                       avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(pos_centerness_pred, pos_centerness_targets, avg_factor=num_pos)

            loss_bs = F.smooth_l1_loss(pos_bs_preds, pos_bs_targets, reduction="none")
            loss_bs = ((loss_bs.mean(dim=-1) * pos_centerness_targets).sum() / centerness_denorm)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness_pred.sum()
            loss_bs = pos_bs_targets.sum()

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_centerness=loss_centerness, loss_bs=loss_bs)

    def get_targets(self, points, gt_bboxes_list, gt_cp_list, gt_labels_list):

        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, bs_targets_list = multi_apply(self._get_target_single,
                                                                      gt_bboxes_list,
                                                                      gt_labels_list,
                                                                      gt_cp_list,
                                                                      points=concat_points,
                                                                      regress_ranges=concat_regress_ranges,
                                                                      num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [bbox_targets.split(num_points, 0) for bbox_targets in bbox_targets_list]
        bs_targets_list = [bs_targets.split(num_points, 0) for bs_targets in bs_targets_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_bs_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list])
            bs_targets = torch.cat([bs_targets[i] for bs_targets in bs_targets_list])

            bbox_targets = bbox_targets / self.strides[i]
            bs_targets = bs_targets / self.strides[i]

            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_bs_targets.append(bs_targets)

    def _get_target_single(self, gt_bboxes, gt_labels, gt_cp, points, regress_ranges, num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)),\
                   gt_cp.new_zeros((num_points, 16))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)

        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        gt_cp = gt_cp[None].expand(num_points, num_gts, int(self.cp_num[0] * 2)).clone()

        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        xs_cp = xs.repeat((1, int(self.cp_num[0]))).reshape(len(xs), -1, int(len(gt_cp[0][0]) / 2))
        ys_cp = ys.repeat((1, int(self.cp_num[0]))).reshape(len(ys), -1, int(len(gt_cp[0][0]) / 2))

        gt_cp[:, :, ::2] = gt_cp[:, :, ::2] - xs_cp
        gt_cp[:, :, 1::2] = gt_cp[:, :, 1::2] - ys_cp

        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = ((max_regress_distance >= regress_ranges[..., 0])
                                & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        bs_targets = gt_cp[range(num_points), min_area_inds]

        return labels, bbox_targets, bs_targets

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