'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2021-12-21 19:54:22
LastEditors: Zhangzixu
LastEditTime: 2022-01-04 10:36:20
'''
# Copyright (c) OpenMMLab. All rights reserved.
from .base_textdet_targets import BaseTextDetTargets
from .dbnet_targets import DBNetTargets
from .drrg_targets import DRRGTargets
from .fcenet_targets import FCENetTargets
from .panet_targets import PANetTargets
from .psenet_targets import PSENetTargets
from .textsnake_targets import TextSnakeTargets
from .bsnet_targets import BSNetTargets
from .bsnet_targets_bbox import BSNetTargets_bbox
from .wlnet_targets import WLNetTargets
from .bsnet_fcos_targets import BSNetFcosTargets
from .wlnet_fcos_targets import WLNetFcosTargets
from .bsnet_targets_out5 import BSNetTargets_Out5
from .hybridnet_targets import HybridNetTargets
from .fcenet_fcos_targets import FCENetFcosTargets
__all__ = [
    'BaseTextDetTargets', 'PANetTargets', 'PSENetTargets', 'DBNetTargets', 'FCENetTargets', 'TextSnakeTargets', 'DRRGTargets',
    'BSNetTargets', 'BSNetTargets_bbox', 'WLNetTargets', 'BSNetFcosTargets', 'WLNetFcosTargets', 'BSNetTargets_Out5', 'HybridNetTargets', 'FCENetFcosTargets'
]
