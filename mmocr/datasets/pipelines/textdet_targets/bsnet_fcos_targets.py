# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES
from numpy.fft import fft
from numpy.linalg import norm
from mmdet.core import BitmapMasks, PolygonMasks
import mmocr.utils.check_argument as check_argument
from .textsnake_targets import TextSnakeTargets
from .B_Spline import BS_curve
from shapely.geometry.polygon import LinearRing
import mmcv
import cv2


@PIPELINES.register_module()
class BSNetFcosTargets(TextSnakeTargets):
    """Generate the ground truth targets of FCENet: Fourier Contour Embedding
    for Arbitrary-Shaped Text Detection.

    [https://arxiv.org/abs/2104.10442]

    Args:
        fourier_degree (int): The maximum Fourier transform degree k.
        resample_step (float): The step size for resampling the text center
            line (TCL). It's better not to exceed half of the minimum width.
        center_region_shrink_ratio (float): The shrink ratio of text center
            region.
        level_size_divisors (tuple(int)): The downsample ratio on each level.
        level_proportion_range (tuple(tuple(int))): The range of text sizes
            assigned to each level.
    """
    def __init__(self, bs=4, cp=14, resample_num=100):

        super().__init__()

        self.bs_degree = bs
        self.cp_num = cp
        self.resample_num = resample_num

    def normalize_polygon(self, polygon):
        """Normalize one polygon so that its start point is at right most.

        Args:
            polygon (list[float]): The origin polygon.
        Returns:
            new_polygon (lost[float]): The polygon with start point at right.
        """
        if self.flip_flag1 ^ self.flip_flag2:
            polygon = np.flipud(polygon)
        index = np.argsort(polygon[:, 0])[0]
        new_polygon = np.concatenate([polygon[index:], polygon[:index]])
        return new_polygon

    def Resample(self, points, ResampleNum):
        perimeter = LinearRing(points).length
        ResamplePoints = np.empty([0, 2], dtype=np.int32)
        #计算每条边应分得的点数 这里存在一个问题 int()的过程中会使得重采样的点小于ResampleNum 这里采用的策略是将缺少的点分给长的边
        eachLengthPoints = []
        for i, point in enumerate(points):
            try:
                nextPoint = points[i + 1]
            except:
                nextPoint = points[0]
            eachLengthPoints.append(int(np.linalg.norm((point - nextPoint)) * ResampleNum / perimeter))

        eachLengthPoints = np.array(eachLengthPoints)
        # print(eachLengthPoints.sum())
        if eachLengthPoints.sum() < 100:
            lostPoints = ResampleNum - eachLengthPoints.sum()
            index = np.arange(len(eachLengthPoints))
            total = np.column_stack((index, eachLengthPoints))
            total = total[np.argsort(total[:, 1])]

            Temp = np.zeros_like(eachLengthPoints)
            Temp[-lostPoints:] = 1
            total[:, 1] += Temp
            total = total[np.argsort(total[:, 0])]

            eachLengthPoints = total[:, 1]
        elif eachLengthPoints.sum() > 100:
            lostPoints = eachLengthPoints.sum() - ResampleNum
            Temp = np.zeros_like(eachLengthPoints)
            Temp[0:lostPoints] = 1
            eachLengthPoints += Temp

        else:
            pass
        if eachLengthPoints.sum() != ResampleNum:
            raise ValueError("重采样点数不符")
        #eachLengthPoints中存放着每条边应当重采样的点的数目
        for i, point in enumerate(points):
            try:
                nextPoint = points[i + 1]
            except:
                nextPoint = points[0]
            sectionPoints = np.linspace(point, nextPoint, eachLengthPoints[i])
            ResamplePoints = np.append(ResamplePoints, sectionPoints, axis=0)

        return ResamplePoints

    def cal_cp_coordinates(self, polygon, bs_degree, cp_num, resample_num):
        ResamplePoints = self.Resample(polygon, resample_num)
        ResamplePoints = np.vstack((ResamplePoints, ResamplePoints[0]))
        bs = BS_curve(cp_num, bs_degree)
        paras = bs.estimate_parameters(ResamplePoints)
        knots = bs.get_knots()
        if bs.check():
            cp = bs.approximation(ResamplePoints)
        CtrlPoints = cp[:-1]

        return CtrlPoints.flatten().astype(np.int32)

    def cal_contour_points(self, bs_cp):
        bs_cp_int = bs_cp.reshape((-1, 2))  #.astype(np.int32)
        bs_cp_int = np.append(bs_cp_int, bs_cp_int[0]).reshape((-1, 2))
        bs = BS_curve(self.cp_num, 4, cp=bs_cp_int)
        uq = np.linspace(0, 1, 51)
        bs.set_paras(uq)
        knots = bs.get_knots()

        points = bs.bs(uq)
        points = np.array(points).astype(np.int32)
        points = points[:-1]
        return points

    def generate_gt_targets(self, img, text_polys, ignore_polys):
        """Generate ground truth target on each level.

        Args:
            img_size (list[int]): Shape of input image.
            text_polys (list[list[ndarray]]): A list of ground truth polygons.
            ignore_polys (list[list[ndarray]]): A list of ignored polygons.
        Returns:
            level_maps (list(ndarray)): A list of ground target on each level.
        """
        # h, w = img_size
        gt_bboxes = np.zeros((len(text_polys), 4), dtype=np.float32)
        gt_bs_cp = np.zeros((len(text_polys), self.cp_num * 2), dtype=np.float32)
        gt_lables = np.zeros(len(text_polys), dtype=np.float32)
        # gt_bboxes = np.array([])
        # gt_bs_cp = np.array([])
        for j, poly in enumerate(text_polys):
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]] for i in range(0, len(poly[0]), 2)]
            polygon = np.array(text_instance, dtype=np.int).reshape((1, -1, 2))
            polygon = self.normalize_polygon(polygon[0])
            box_x, box_y, box_w, box_h = cv2.boundingRect(polygon)
            single_bs_cp = self.cal_cp_coordinates(polygon, self.bs_degree, self.cp_num, self.resample_num)
            gt_bboxes[j] = np.array([box_x, box_y, box_x + box_w, box_y + box_h])
            # single_bs_cp = self.normalize_polygon(single_bs_cp.reshape(-1, 2), flip_flag).flatten()
            gt_bs_cp[j] = np.array(single_bs_cp)

            # polygon_color = mmcv.color_val('green')
            # cp_color = mmcv.color_val('red')

            # reconstrucationPoints = self.cal_contour_points(single_bs_cp)
            # # cv2.polylines(img, [polygon.reshape(-1, 1, 2)], True, color=polygon_color, thickness=2)
            # cv2.polylines(img, [reconstrucationPoints.reshape(-1, 1, 2)], True, color=polygon_color, thickness=2)
            # # cv2.polylines(img, [single_bs_cp.reshape(-1, 1, 2)], True, color=cp_color, thickness=2)
            # cv2.polylines(img, [polygon.reshape(-1, 1, 2)], True, color=cp_color, thickness=2)
            # cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 0, 0), thickness=2)
            # mmcv.imwrite(img, 'img.jpg')
        return gt_bboxes, gt_bs_cp, gt_lables

    def generate_targets(self, results):
        """Generate the ground truth targets for FCENet.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """

        assert isinstance(results, dict)
        img = results['img']

        self.flip_flag1 = results['Stage1_flip']
        self.flip_flag2 = results['flip']

        polygon_masks = results['gt_masks'].masks
        polygon_masks_ignore = results['gt_masks_ignore'].masks

        h, w, _ = results['img_shape']

        gt_bboxes, gt_bs_cp, gt_lables = self.generate_gt_targets(img, polygon_masks, polygon_masks_ignore)

        mapping = {'gt_bboxes': gt_bboxes, 'gt_cp': gt_bs_cp, 'gt_lables': gt_lables}
        for key, value in mapping.items():
            results[key] = value

        return results