'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2021-12-21 19:54:22
LastEditors: Zhangzixu
LastEditTime: 2022-01-07 11:52:16
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
from .bs_head_fcos_tb import BS_FCOSHead_tb
from .hybrid_head_tb import HybridHead_tb
from .bs_head_tood import BS_TOODHead
from .bs_head_tb import BSHead_tb
from .bs_head_tb_icdar import BSHead_tb_icdar
from .wl_head_diff_weight import WLHead_diff_weight
from .wl_head_direct import WLHead_direct
from .wl_head_chamfer import WLHead_chamfer

__all__ = [
    'PSEHead', 'PANHead', 'DBHead', 'FCEHead', 'TextSnakeHead', 'DRRGHead', 'HeadMixin', 'BSHead', 'BSHead_BBOXES', 'WLHead',
    'BS_FCOSHead', 'WLHead_fcos', 'BS_FCOSHead_Att', 'HybridHead', 'FCE_FCOSHead', 'BS_FCOSHead_tb', 'HybridHead_tb',
    'BS_TOODHead', 'BSHead_tb', "BSHead_tb_icdar", 'WLHead_diff_weight', 'WLHead_direct', 'WLHead_chamfer'
]
