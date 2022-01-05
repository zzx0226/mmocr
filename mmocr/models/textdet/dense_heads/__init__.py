'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2021-12-21 19:54:22
LastEditors: Zhangzixu
LastEditTime: 2022-01-04 10:47:41
'''
# Copyright (c) OpenMMLab. All rights reserved.
from .db_head import DBHead
from .drrg_head import DRRGHead
from .fce_head import FCEHead
from .head_mixin import HeadMixin
from .pan_head import PANHead
from .pse_head import PSEHead
from .textsnake_head import TextSnakeHead
from .bs_head import BSHead
from .bs_head_bboxes import BSHead_BBOXES
from .wl_head import WLHead
from .wl_head_fcos import WLHead_fcos
from .bs_head_fcos import BS_FCOSHead
from .bs_head_fcos_attention import BS_FCOSHead_Att
from .hybrid_head import HybridHead
from .fce_head_fcos import FCE_FCOSHead
__all__ = [
    'PSEHead', 'PANHead', 'DBHead', 'FCEHead', 'TextSnakeHead', 'DRRGHead', 'HeadMixin', 'BSHead', 'BSHead_BBOXES', 'WLHead',
    'BS_FCOSHead', 'WLHead_fcos', 'BS_FCOSHead_Att', 'HybridHead', 'FCE_FCOSHead'
]
