'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2022-01-04 10:48:23
LastEditors: Zhangzixu
LastEditTime: 2022-01-04 10:48:55
'''
# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from mmocr.models.builder import DETECTORS, build_backbone, build_neck, build_head
# from .base import BaseDetector
from .base import BaseDetector
from mmdet.core import bbox2result
from .text_detector_mixin import TextDetectorMixin


@DETECTORS.register_module()
class FCENet_fcos(BaseDetector):  #
    """The class for implementing FCENet text detector
    FCENet(CVPR2021): Fourier Contour Embedding for Arbitrary-shaped Text
        Detection

    [https://arxiv.org/abs/2104.10442]
    """

    def __init__(self, backbone, neck=None, bbox_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):

        # BaseDetector.__init__(init_cfg)
        # TextDetectorMixin.__init__(self, show_score)
        # super(BSNet_fcos, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)
        super(FCENet_fcos, self).__init__(init_cfg)
        if pretrained:
            warnings.warn(
                'DeprecationWarning: pretrained is deprecated, ' 'please use "init_cfg" instead')
            backbone.pretrained = pretrained

        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_fourier, gt_bboxes_ignore=None):

        super(FCENet_fcos, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(
            x, img_metas, gt_bboxes, gt_fourier, gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        boundaries = self.bbox_head.get_boundary(*outs, img_metas, rescale)

        return [boundaries]

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
