# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.core import multi_apply, bbox2result
from mmocr.models.builder import HEADS
from ..postprocess.utils import poly_nms
from .head_mixin import HeadMixin


@HEADS.register_module()
class HybridHead(HeadMixin, BaseModule):
    """The class for implementing FCENet head.

    FCENet(CVPR2021): `Fourier Contour Embedding for Arbitrary-shaped Text
    Detection <https://arxiv.org/abs/2104.10442>`_

    Args:
        in_channels (int): The number of input channels.
        scales (list[int]) : The scale of each layer.
        fourier_degree (int) : The maximum Fourier transform degree k.
        nms_thr (float) : The threshold of nms.
        loss (dict): Config of loss for FCENet.
        postprocessor (dict): Config of postprocessor for FCENet.
    """

    def __init__(self,
                 in_channels,
                 scales,
                 bs_degree=4,
                 cp_num=8,
                 fourier_degree=5,
                 nms_thr=0.1,
                 loss=dict(type='HybridLoss', num_sample=50),
                 postprocessor=dict(type='HybridPostprocessor',
                                    text_repr_type='poly',
                                    num_reconstr_points=50,
                                    alpha=1.0,
                                    beta=2.0,
                                    score_thr=0.3),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(type='Normal', mean=0, std=0.01, override=[dict(name='out_conv_cls'),
                                                                          dict(name='out_conv_reg')]),
                 **kwargs):
        old_keys = ['text_repr_type', 'decoding_type',
                    'num_reconstr_points', 'alpha', 'beta', 'score_thr']
        for key in old_keys:
            if kwargs.get(key, None):
                postprocessor[key] = kwargs.get(key)
                warnings.warn(
                    f'{key} is deprecated, please specify '
                    'it in postprocessor config dict. See '
                    'https://github.com/open-mmlab/mmocr/pull/640'
                    ' for details.', UserWarning)
        if kwargs.get('num_sample', None):
            loss['num_sample'] = kwargs.get('num_sample')
            warnings.warn(
                'num_sample is deprecated, please specify '
                'it in loss config dict. See '
                'https://github.com/open-mmlab/mmocr/pull/640'
                ' for details.', UserWarning)
        BaseModule.__init__(self, init_cfg=init_cfg)
        loss['fourier_degree'] = fourier_degree
        loss['bs_degree'] = bs_degree
        loss['cp_num'] = cp_num

        postprocessor['fourier_degree'] = fourier_degree
        postprocessor['bs_degree'] = bs_degree
        postprocessor['cp_num'] = cp_num
        postprocessor['nms_thr'] = nms_thr
        HeadMixin.__init__(self, loss, postprocessor)

        assert isinstance(in_channels, int)

        self.downsample_ratio = 1.0
        self.in_channels = in_channels
        self.scales = scales
        self.fourier_degree = fourier_degree

        self.nms_thr = nms_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.out_channels_cls = 4
        self.out_channels_reg = (2 * self.fourier_degree + 1) * 2

        self.bs_reg = cp_num * 2

        self.out_conv_cls = nn.Conv2d(
            self.in_channels, self.out_channels_cls, kernel_size=3, stride=1, padding=1)
        self.out_conv_reg = nn.Conv2d(
            self.in_channels, self.out_channels_reg, kernel_size=3, stride=1, padding=1)
        self.bs_conv_reg = nn.Conv2d(
            self.in_channels, self.bs_reg, kernel_size=3, stride=1, padding=1)

    def forward(self, feats):
        """
        Args:
            feats (list[Tensor]): Each tensor has the shape of :math:`(N, C_i,
                H_i, W_i)`.

        Returns:
            list[[Tensor, Tensor]]: Each pair of tensors corresponds to the
            classification result and regression result computed from the input
            tensor with the same index. They have the shapes of :math:`(N,
            C_{cls,i}, H_i, W_i)` and :math:`(N, C_{out,i}, H_i, W_i)`.
        """
        cls_res, reg_res, bs_res = multi_apply(self.forward_single, feats)
        level_num = len(cls_res)
        preds = [[cls_res[i], reg_res[i], bs_res[i]] for i in range(level_num)]
        return preds

    def forward_single(self, x):
        cls_predict = self.out_conv_cls(x)
        reg_predict = self.out_conv_reg(x)
        bs_predict = self.bs_conv_reg(x)
        return cls_predict, reg_predict, bs_predict

    def get_boundary(self, score_maps, img_metas, rescale):
        assert len(score_maps) == len(self.scales)

        boundaries = []
        for idx, score_map in enumerate(score_maps):
            scale = self.scales[idx]
            boundaries = boundaries + \
                self._get_boundary_single(score_map, scale)

        # nms
        boundaries = poly_nms(boundaries, self.nms_thr)

        if rescale:
            boundaries = self.resize_boundary(
                boundaries, 1.0 / img_metas[0]['scale_factor'])

        results = dict(boundary_result=boundaries)
        return results

    def _get_boundary_single(self, score_map, scale):
        assert len(score_map) == 3
        assert score_map[1].shape[1] == 4 * self.fourier_degree + 2

        return self.postprocessor(score_map, scale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes) for det_bboxes, det_labels in results_list
        ]
        return bbox_results
