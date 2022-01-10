# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import DETECTORS
from .single_stage import SingleStageDetectorTood


@DETECTORS.register_module()
class BSNet_TOOD(SingleStageDetectorTood):
    r"""Implementation of `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_."""
    def __init__(self, backbone, neck, bbox_head, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        super(BSNet_TOOD, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)

    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch
