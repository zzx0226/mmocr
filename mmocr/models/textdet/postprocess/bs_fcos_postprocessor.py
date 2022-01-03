# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import torch
from mmocr.models.builder import POSTPROCESSOR
from .base_postprocessor import BasePostprocessor
from .utils import poly_nms
from .B_Spline import BS_curve

from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl


@POSTPROCESSOR.register_module()
class BSFcosPostprocessor(BasePostprocessor):
    """Decoding predictions of FCENet to instances.

    Args:
        fourier_degree (int): The maximum Fourier transform degree k.
        num_reconstr_points (int): The points number of the polygon
            reconstructed from predicted Fourier coefficients.
        text_repr_type (str): Boundary encoding type 'poly' or 'quad'.
        scale (int): The down-sample scale of the prediction.
        alpha (float): The parameter to calculate final scores. Score_{final}
                = (Score_{text region} ^ alpha)
                * (Score_{text center region}^ beta)
        beta (float): The parameter to calculate final score.
        score_thr (float): The threshold used to filter out the final
            candidates.
        nms_thr (float): The threshold of nms.
    """
    def __init__(
            self,
            #  fourier_degree,
            bs_degree,
            cp_num,
            num_reconstr_points,
            # text_repr_type='poly',
            alpha=1.0,
            beta=2.0,
            score_thr=0.3,
            nms_thr=0.1,
            **kwargs):
        # super().__init__(text_repr_type)
        # self.fourier_degree = fourier_degree
        self.bs_degree = bs_degree
        self.cp_num = cp_num

        self.num_reconstr_points = num_reconstr_points
        self.alpha = alpha
        self.beta = beta
        self.score_thr = score_thr
        self.nms_thr = nms_thr

    def __call__(self,
                 cls_scores,
                 bbox_preds,
                 bs_preds,
                 score_factors=None,
                 img_metas=None,
                 rescale=False,
                 cfg=None,
                 with_nms=True,
                 **kwargs):
        assert len(cls_scores) == len(bbox_preds) == len(score_factors) == len(bs_preds)

        cfg = self.test_cfg
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)

        All_Results = []
        result_list = []
        # bs_list = []
        # bbox_list = []
        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            bs_pred_list = select_single_mlvl(bs_preds, img_id)

            score_factor_list = select_single_mlvl(score_factors, img_id)

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list, bs_pred_list, score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms, **kwargs)

            bbox_temp, lable_temp, bs_temp = results
            bboxes = np.vstack(bbox_temp.cpu().numpy())
            bs_cps = np.vstack(bs_temp.cpu().numpy())
            scores = bboxes[:, -1]

            All_Results.append(results)

            for i, bs_cp in enumerate(bs_cps):

                bs_cp_int = bs_cp.reshape((-1, 2))
                bs_cp_int = np.append(bs_cp_int, bs_cp_int[0]).reshape((-1, 2))
                bs = BS_curve(int(bs_cps.shape[1] / 2), 4, cp=bs_cp_int)
                uq = np.linspace(0, 1, 51)
                bs.set_paras(uq)
                knots = bs.get_knots()

                points = bs.bs(uq)
                points = np.array(points)  #.astype(np.int32)
                points = points[:-1]
                points = np.append(points, scores[i])

                result_list.append(points.tolist())
        # boundaries = poly_nms(boundaries, self.nms_thr)
        result_list = poly_nms(result_list, self.nms_thr)
        results = dict(boundary_result=result_list)
        return results

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           cp_preds_list,
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
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, cp_pred,score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,cp_preds_list,
                              score_factor_list, mlvl_priors)):
            # for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
            #         enumerate(zip(cls_score_list, bbox_pred_list,
            #                       score_factor_list, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            cp_pred = cp_pred.permute(1, 2, 0).reshape(-1, 16)

            if with_score_factors:
                score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
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
            results = filter_scores_and_topk(scores, cfg.score_thr, nms_pre,
                                             dict(bbox_pred=bbox_pred, cp_pred=cp_pred, priors=priors))
            # results = filter_scores_and_topk(scores, cfg.score_thr, nms_pre, dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            cp_pred = filtered_results['cp_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes, img_meta['scale_factor'], cfg, rescale, with_nms,
                                       mlvl_score_factors, **kwargs)

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
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels) == len(mlvl_bs)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bs = torch.cat(mlvl_bs)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bs /= mlvl_bs.new_tensor(np.tile(np.array(scale_factor), int(mlvl_bs.shape[1] / 4)))
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

            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores, mlvl_labels, cfg.nms)
            #det_bs, keep_idxs = batched_nms(mlvl_bs, mlvl_scores, mlvl_labels, cfg.nms)

            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]

            det_bs = mlvl_bs[keep_idxs][:cfg.max_per_img]
            return det_bboxes, det_labels, det_bs
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_labels, mlvl_bs