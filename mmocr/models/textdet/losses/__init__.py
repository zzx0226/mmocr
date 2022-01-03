'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2021-12-21 19:54:22
LastEditors: Zhangzixu
LastEditTime: 2022-01-03 13:18:21
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
__all__ = [
    'PANLoss', 'PSELoss', 'DBLoss', 'TextSnakeLoss', 'FCELoss', 'DRRGLoss', 'BSLoss', 'BSLoss_bbox', 'WLLoss', 'BSLoss_fcos', 'BSLoss_Out5', 'HybridLoss'
]
