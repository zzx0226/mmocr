import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class Load_wl_coeffs():
    def __init__(self):
        pass

    def __call__(self, results):
        results['wl_haar'] = results['ann_info']['wl_haar']

        return results