import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class Load_wl():
    def __init__(self):
        pass

    def __call__(self, results):
        results['gt_wl'] = results['ann_info']['']

        return results