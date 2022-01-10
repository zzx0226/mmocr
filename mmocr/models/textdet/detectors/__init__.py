'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2021-12-21 19:54:22
LastEditors: Zhangzixu
LastEditTime: 2022-01-04 10:49:18
'''
# Copyright (c) OpenMMLab. All rights reserved.
from .dbnet import DBNet
from .drrg import DRRG
from .fcenet import FCENet
from .ocr_mask_rcnn import OCRMaskRCNN
from .panet import PANet
from .psenet import PSENet
from .single_stage_text_detector import SingleStageTextDetector
from .text_detector_mixin import TextDetectorMixin
from .textsnake import TextSnake
from .bsnet import BSNet
from .wlnet import WLNet
from .wlnet_fcos import WLNet_FCOS
from .bsnet_fcos import BSNet_fcos
from .hybridnet import HybridNet
from .fcenet_fcos import FCENet_fcos
from .bsnet_tood import BSNet_TOOD
from .single_stage import SingleStageDetectorTood

__all__ = [
    'TextDetectorMixin', 'SingleStageTextDetector', 'OCRMaskRCNN', 'DBNet', 'PANet', 'PSENet', 'TextSnake', 'FCENet', 'DRRG',
    'BSNet', 'WLNet', 'BSNet_fcos', 'WLNet_FCOS', 'HybridNet', 'FCENet_fcos', 'BSNet_TOOD', 'SingleStageDetectorTood'
]
