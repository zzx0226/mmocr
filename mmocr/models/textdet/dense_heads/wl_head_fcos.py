# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.core import multi_apply, reduce_mean
from mmdet.models.builder import build_loss
from mmocr.models.builder import HEADS
from ..postprocess.utils import poly_nms
from .head_mixin import HeadMixin
from mmcv.cnn import Scale
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
import torch.nn.functional as F
import numpy as np
from mmcv.ops import batched_nms
import pywt

INF = 1e8


@HEADS.register_module()
class WLHead_fcos(AnchorFreeHead, HeadMixin):
    def __init__(
            self,
            wavelet_type='haar',
            num_classes=1,
            in_channels=256,
            strides=[8, 16, 32, 64, 128],
            regress_ranges=((-1, 64), (64, 128), (128, 256),
                            (256, 512), (512, INF)),
            nms_thr=0.3,
            center_sampling=True,
            center_sample_radius=1.5,
            # loss=dict(type='BSLoss_fcos',
            loss_cls=dict(type='FocalLoss', use_sigmoid=True,
                          gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_bbox=dict(type='IoULoss', loss_weight=1.0),
            loss_centerness=dict(type='CrossEntropyLoss',
                                 use_sigmoid=True, loss_weight=1.0),
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            init_cfg=dict(type='Normal',
                          layer='Conv2d',
                          std=0.01,
                          override=dict(type='Normal', name='conv_cls', std=0.01, bias_prob=0.01)),
            **kwargs):

        super().__init__(num_classes,
                         in_channels,
                         strides=strides,
                         loss_cls=loss_cls,
                         loss_bbox=loss_bbox,
                         norm_cfg=norm_cfg,
                         init_cfg=init_cfg,
                         **kwargs)
        self.wavelet_type = wavelet_type
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_centerness = build_loss(loss_centerness)
        self.conv_wl = nn.Conv2d(self.feat_channels, 28, 3, padding=1)
        self.nms_thr = nms_thr
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales, self.strides)

    def forward_single_scale(self, x):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.

        Returns:
            tuple: Scores for each class, bbox predictions, features
                after classification and regression conv layers, some
                models needs these features like FCOS.
        """
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)

        bbox_pred = self.conv_reg(reg_feat)
        wl_pred = self.conv_wl(reg_feat)

        centerness = self.conv_centerness(reg_feat)

        bbox_pred = F.relu(bbox_pred)
        # bs_pred = F.relu(bs_pred)

        return cls_score, bbox_pred, wl_pred, centerness

    def forward_single(self, x, scale, stride):

        cls_score, bbox_pred, wl_pred, centerness = self.forward_single_scale(
            x)  # , bs_feat
        bbox_pred = scale(bbox_pred).float()
        wl_pred = scale(wl_pred).float()

        if not self.training:
            bbox_pred *= stride
            wl_pred *= stride

        return cls_score, bbox_pred, wl_pred, centerness

    def get_boundary(self,
                     cls_scores,
                     bbox_preds,
                     wl_preds,
                     score_factors=None,
                     img_metas=None,
                     rescale=False,
                     cfg=None,
                     with_nms=True,
                     **kwargs):
        assert len(cls_scores) == len(bbox_preds) == len(
            score_factors) == len(wl_preds)

        cfg = self.test_cfg
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)

        result_list = []
        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            wl_pred_list = select_single_mlvl(wl_preds, img_id)

            score_factor_list = select_single_mlvl(score_factors, img_id)

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list, wl_pred_list, score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms, **kwargs)

            bbox_temp, lable_temp, wl_temp = results
            bboxes = np.vstack(bbox_temp.cpu().numpy())
            wl_coeffs = np.vstack(wl_temp.cpu().numpy())
            scores = bboxes[:, -1]

        result_list = poly_nms(result_list, self.nms_thr)

        # if rescale:
        #     result_list = self.resize_boundary(result_list, 1.0 / img_metas[0]['scale_factor'])

        results = dict(boundary_result=result_list)
        return results

    def _get_boundary_single(self, score_map, scale):
        assert len(score_map) == 2
        assert score_map[1].shape[1] == 28

        return self.postprocessor(score_map, scale)

    def loss(self,
             cls_scores,
             bbox_preds,
             wl_preds,
             centernesses,
             gt_bboxes,
             gt_wl,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        assert len(cls_scores) == len(bbox_preds) == len(
            centernesses) == len(wl_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(featmap_sizes,
                                                            dtype=bbox_preds[0].dtype,
                                                            device=bbox_preds[0].device)
        labels, bbox_targets, wl_targets = self.get_targets(
            all_level_points, gt_bboxes, gt_wl, gt_labels)
        # , cp_target
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [cls_score.permute(
            0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(
            0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds]
        flatten_centerness_pred = [centerness.permute(
            0, 2, 3, 1).reshape(-1) for centerness in centernesses]

        flatten_wl_preds = [wl_pred.permute(
            0, 2, 3, 1).reshape(-1, 28) for wl_pred in wl_preds]

        # flatten_wl_preds = flatten_wl_preds.cpu().detach().numpy().flatten()
        for i, flatten_wl_pred in enumerate(flatten_wl_preds):
            flatten_wl_pred = flatten_wl_pred.cpu().detach().numpy().flatten()
            num = int(len(flatten_wl_pred) / 4)
            cA = flatten_wl_pred[0:num].reshape(-1, 1)
            cH = flatten_wl_pred[num:num * 2].reshape(-1, 1)
            cV = flatten_wl_pred[num * 2:num * 3].reshape(-1, 1)
            cD = flatten_wl_pred[num * 3:num * 4].reshape(-1, 1)
            reconstructs = pywt.waverec2(
                (cA, (cH, cV, cD)), 'haar').reshape(-1, 28)
            flatten_wl_pred = torch.tensor(reconstructs).cuda()
            flatten_wl_preds[i] = flatten_wl_pred

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness_pred = torch.cat(flatten_centerness_pred)

        flatten_wl_preds = torch.cat(flatten_wl_preds)

        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)

        flatten_wl_targets = torch.cat(wl_targets)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0) & (
            flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels.long(), avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_wl_preds = flatten_wl_preds[pos_inds]

        pos_centerness_pred = flatten_centerness_pred[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]

        pos_wl_targets = flatten_wl_targets[pos_inds]

        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(reduce_mean(
            pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(pos_decoded_bbox_preds,
                                       pos_decoded_target_preds,
                                       weight=pos_centerness_targets,
                                       avg_factor=centerness_denorm)

            loss_centerness = self.loss_centerness(
                pos_centerness_pred, pos_centerness_targets, avg_factor=num_pos)

            loss_wl = F.smooth_l1_loss(
                pos_wl_preds, pos_wl_targets, reduction="none")
            loss_wl = ((loss_wl.mean(dim=-1) *
                       pos_centerness_targets).sum() / centerness_denorm)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness_pred.sum()
            loss_wl = pos_wl_targets.sum()

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_centerness=loss_centerness, loss_wl=loss_wl)

    def get_targets(self, points, gt_bboxes_list, gt_wl_list, gt_labels_list):

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
        labels_list, bbox_targets_list, wl_targets_list = multi_apply(self._get_target_single,
                                                                      gt_bboxes_list,
                                                                      gt_labels_list,
                                                                      gt_wl_list,
                                                                      points=concat_points,
                                                                      regress_ranges=concat_regress_ranges,
                                                                      num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [bbox_targets.split(
            num_points, 0) for bbox_targets in bbox_targets_list]
        wl_targets_list = [wl_targets.split(
            num_points, 0) for wl_targets in wl_targets_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_wl_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat([bbox_targets[i]
                                     for bbox_targets in bbox_targets_list])
            wl_targets = torch.cat([wl_targets[i]
                                   for wl_targets in wl_targets_list])

            bbox_targets = bbox_targets / self.strides[i]
            wl_targets = wl_targets / self.strides[i]

            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_wl_targets.append(wl_targets)

        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_wl_targets

    def _get_target_single(self, gt_bboxes, gt_labels, gt_wl, points, regress_ranges, num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_bboxes.size(0)
        gt_labels = gt_labels[:num_gts]
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                gt_bboxes.new_zeros((num_points, 4)),\
                gt_wl.new_zeros((num_points, 28))
        # wl = torch.zeros((num_gts, 28))
        # num = 7
        # for i, each_wl in enumerate(gt_wl):
        #     each_wl = each_wl.cpu().detach().numpy()
        #     cA = each_wl[0:num].reshape(-1, 1)
        #     cH = each_wl[num:num * 2].reshape(-1, 1)
        #     cV = each_wl[num * 2:num * 3].reshape(-1, 1)
        #     cD = each_wl[num * 3:num * 4].reshape(-1, 1)
        #     reconstructs = pywt.waverec2((cA, (cH, cV, cD)), 'haar').flatten()
        #     wl[i] = torch.tensor(reconstructs)
        # gt_wl = wl.cuda()
        cat_wl = gt_wl.cpu().detach().numpy().flatten()
        num = int(gt_wl.shape[0] * gt_wl.shape[1] / 4)
        cA = cat_wl[0:num].reshape(-1, 1)
        cH = cat_wl[num:num * 2].reshape(-1, 1)
        cV = cat_wl[num * 2:num * 3].reshape(-1, 1)
        cD = cat_wl[num * 3:num * 4].reshape(-1, 1)
        reconstructs = pywt.waverec2(
            (cA, (cH, cV, cD)), 'haar').reshape(num_gts, -1)
        gt_wl = torch.tensor(reconstructs).cuda()

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * \
            (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)

        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        gt_wl = gt_wl[None].expand(num_points, num_gts, 28).clone()

        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        xs_cp = xs.repeat((1, 14)).reshape(
            len(xs), -1, int(len(gt_wl[0][0]) / 2))
        ys_cp = ys.repeat((1, 14)).reshape(
            len(ys), -1, int(len(gt_wl[0][0]) / 2))

        gt_wl[:, :, ::2] = gt_wl[:, :, ::2] - xs_cp
        gt_wl[:, :, 1::2] = gt_wl[:, :, 1::2] - ys_cp

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(
                x_mins > gt_bboxes[..., 0], x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(
                y_mins > gt_bboxes[..., 1], y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(
                x_maxs > gt_bboxes[..., 2], gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(
                y_maxs > gt_bboxes[..., 3], gt_bboxes[..., 3], y_maxs)
            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
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
        wl_targets = gt_wl[range(num_points), min_area_inds]

        return labels, bbox_targets, wl_targets

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

    def get_boundary(self,
                     cls_scores,
                     bbox_preds,
                     wl_preds,
                     score_factors=None,
                     img_metas=None,
                     rescale=False,
                     cfg=None,
                     with_nms=True,
                     **kwargs):
        assert len(cls_scores) == len(bbox_preds) == len(
            score_factors) == len(wl_preds)
        cfg = self.test_cfg
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)

        flatten_wl_preds = [wl_pred.permute(
            0, 2, 3, 1).reshape(-1, 28) for wl_pred in wl_preds]

        # flatten_wl_preds = flatten_wl_preds.cpu().detach().numpy().flatten()
        for i, flatten_wl_pred in enumerate(flatten_wl_preds):
            flatten_wl_pred = flatten_wl_pred.cpu().detach().numpy().flatten()
            num = int(len(flatten_wl_pred) / 4)
            cA = flatten_wl_pred[0:num].reshape(-1, 1)
            cH = flatten_wl_pred[num:num * 2].reshape(-1, 1)
            cV = flatten_wl_pred[num * 2:num * 3].reshape(-1, 1)
            cD = flatten_wl_pred[num * 3:num * 4].reshape(-1, 1)
            reconstructs = pywt.waverec2(
                (cA, (cH, cV, cD)), 'haar').reshape(-1, 28)
            flatten_wl_pred = torch.tensor(reconstructs).cuda()
            flatten_wl_preds[i] = flatten_wl_pred.reshape(
                (wl_preds[i].shape[0], wl_preds[i].shape[1], wl_preds[i].shape[2], wl_preds[i].shape[3]))

        # All_Results = []
        result_list = []
        # bs_list = []
        # bbox_list = []
        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            wl_pred_list = select_single_mlvl(flatten_wl_preds, img_id)

            score_factor_list = select_single_mlvl(score_factors, img_id)

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list, wl_pred_list, score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms, **kwargs)

            bbox_temp, lable_temp, wl_temp = results
            bboxes = np.vstack(bbox_temp.cpu().numpy())
            wl_cps = np.vstack(wl_temp.cpu().numpy())
            scores = bboxes[:, -1]
            for i, wl_cp in enumerate(wl_cps):
                points = np.append(wl_cp, scores[i])
                result_list.append(points.tolist())
            # All_Results.append(results)
            # for bbox in bboxes:
            #     result_list.append(bbox)
            # for i, bs_cp in enumerate(bs_cps):

            #     bs_cp_int = bs_cp.reshape((-1, 2))
            #     bs_cp_int = np.append(bs_cp_int, bs_cp_int[0]).reshape((-1, 2))
            #     bs = BS_curve(int(bs_cps.shape[1] / 2), 4, cp=bs_cp_int)
            #     uq = np.linspace(0, 1, 51)
            #     bs.set_paras(uq)
            #     knots = bs.get_knots()

            #     points = bs.bs(uq)
            #     points = np.array(points)  #.astype(np.int32)
            #     points = points[:-1]
            #     points = np.append(points, scores[i])

            #     result_list.append(points.tolist())

        result_list = poly_nms(result_list, self.nms_thr)
        results = dict(boundary_result=result_list)
        return results

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           bs_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True
        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']

        # cfg.score_thr = 0.05
        nms_pre = cfg.nms_pre

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_bs = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, bs_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, bs_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            bs_pred = bs_pred.permute(1, 2, 0).reshape(-1, bs_pred.shape[0])
            if with_score_factors:
                score_factor = score_factor.permute(
                    1, 2, 0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(
                1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(scores, cfg['score_thr'], nms_pre,
                                             dict(bbox_pred=bbox_pred, bs_pred=bs_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            bs_pred = filtered_results['bs_pred']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            bs_detections = priors.unsqueeze(
                1) + bs_pred.view(-1, int(bs_pred.shape[1] / 2), 2)
            bs_detections = bs_detections.view(-1, bs_pred.shape[1])

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_bs.append(bs_detections)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes, mlvl_bs, img_meta['scale_factor'], cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs)

    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           mlvl_bs,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually with_nms is False is used for aug test.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
           mlvl_labels (list[Tensor]): Box class labels from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
            mlvl_score_factors (list[Tensor], optional): Score factor from
                all scale levels of a single image, each item has shape
                (num_bboxes, ). Default: None.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(
            mlvl_labels) == len(mlvl_bs)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bs = torch.cat(mlvl_bs)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bs /= mlvl_bs.new_tensor(
                np.tile(np.array(scale_factor), int(mlvl_bs.shape[1] / 4)))
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                det_bs = torch.cat([mlvl_bs, mlvl_scores[:, None]], -1)
                return det_bboxes, mlvl_labels, det_bs

            det_bboxes, keep_idxs = batched_nms(
                mlvl_bboxes, mlvl_scores, mlvl_labels, cfg.nms)
            #det_bs, keep_idxs = batched_nms(mlvl_bs, mlvl_scores, mlvl_labels, cfg.nms)

            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]

            det_bs = mlvl_bs[keep_idxs][:cfg.max_per_img]
            return det_bboxes, det_labels, det_bs
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_labels, mlvl_bs
