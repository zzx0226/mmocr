import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC


@PIPELINES.register_module()
class Resize_wl():
    def __init__(self):
        pass

    def __call__(self, results):
        # print("1")  #scale_factor
        scale = np.tile(results['scale_factor'],
                        int(len(results['gt_cp'][0]) / 4 * len(results['gt_cp']))).reshape(len(results['gt_cp']), -1)
        results['gt_cp'] = results['gt_cp'] * scale
        # results['gt_cp'] = DC(results['gt_cp'], cpu_only=True)
        return results