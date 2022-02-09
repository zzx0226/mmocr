'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2021-12-21 19:54:22
LastEditors: Zhangzixu
LastEditTime: 2022-01-07 11:54:09
'''
# Copyright (c) OpenMMLab. All rights reserved.
from .db_loss import DBLoss
from .drrg_loss import DRRGLoss
from .fce_loss import FCELoss
from .pan_loss import PANLoss
from .pse_loss import PSELoss
from .textsnake_loss import TextSnakeLoss
from .bs_loss import BSLoss
from .bs_loss_bbox import BSLoss_bbox
from .wl_loss import WLLoss
from .bs_loss_fcos import BSLoss_fcos
from .bs_loss_out5 import BSLoss_Out5
from .hybrid_loss import HybridLoss
from .hybrid_loss_tb import HybridLoss_tb
from .bs_loss_tb import BSLoss_tb
from .bs_loss_tb_new import BSLoss_tb_new
from .chamfer_distance import ChamferDistance, chamfer_distance
from .bs_loss_tb_icdar import BSLoss_tb_icdar
from .wl_loss_test import WLLoss_test

__all__ = [
    'PANLoss', 'PSELoss', 'DBLoss', 'TextSnakeLoss', 'FCELoss', 'DRRGLoss', 'BSLoss', 'BSLoss_bbox', 'WLLoss', 'BSLoss_fcos',
    'BSLoss_Out5', 'HybridLoss', 'HybridLoss_tb', 'BSLoss_tb', 'BSLoss_tb_new', 'ChamferDistance', 'chamfer_distance',
    'BSLoss_tb_icdar', 'WLLoss_test'
]
