import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class Load_bs_cp():
    def __init__(self):
        pass

    def __call__(self, results):
        results['gt_cp'] = results['ann_info']['bs_cp']

        return results