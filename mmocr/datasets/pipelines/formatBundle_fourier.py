'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2022-01-04 10:36:31
LastEditors: Zhangzixu
LastEditTime: 2022-01-04 10:36:50
'''
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC


@PIPELINES.register_module()
class FormatBundle_Fourier():
    def __init__(self):
        pass

    def to_tensor(self, data):
        """Convert objects of various python types to :obj:`torch.Tensor`.

        Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
        :class:`Sequence`, :class:`int` and :class:`float`.

        Args:
            data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
                be converted.
        """

        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        else:
            raise TypeError(
                f'type {type(data)} cannot be converted to tensor.')

    def __call__(self, results):
        results['gt_fourier'] = DC(self.to_tensor(results['gt_fourier']))
        return results
