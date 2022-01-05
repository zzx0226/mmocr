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
class FCENetFcosTargets(TextSnakeTargets):
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

    def __init__(self, fourier_degree=5):  # ,bs=4, cp=14, resample_num=100):

        super().__init__()
        self.fourier_degree = fourier_degree
        # self.bs_degree = bs
        # self.cp_num = cp
        # self.resample_num = resample_num

    # def normalize_polygon(self, polygon):
    #     """Normalize one polygon so that its start point is at right most.

    #     Args:
    #         polygon (list[float]): The origin polygon.
    #     Returns:
    #         new_polygon (lost[float]): The polygon with start point at right.
    #     """
    #     if self.flip_flag1 ^ self.flip_flag2:
    #         polygon = np.flipud(polygon)
    #     index = np.argsort(polygon[:, 0])[0]
    #     new_polygon = np.concatenate([polygon[index:], polygon[:index]])
    #     return new_polygon

    # def Resample(self, points, ResampleNum):
    #     perimeter = LinearRing(points).length
    #     ResamplePoints = np.empty([0, 2], dtype=np.int32)
    #     # 计算每条边应分得的点数 这里存在一个问题 int()的过程中会使得重采样的点小于ResampleNum 这里采用的策略是将缺少的点分给长的边
    #     eachLengthPoints = []
    #     for i, point in enumerate(points):
    #         try:
    #             nextPoint = points[i + 1]
    #         except:
    #             nextPoint = points[0]
    #         eachLengthPoints.append(
    #             int(np.linalg.norm((point - nextPoint)) * ResampleNum / perimeter))

    #     eachLengthPoints = np.array(eachLengthPoints)
    #     # print(eachLengthPoints.sum())
    #     if eachLengthPoints.sum() < 100:
    #         lostPoints = ResampleNum - eachLengthPoints.sum()
    #         index = np.arange(len(eachLengthPoints))
    #         total = np.column_stack((index, eachLengthPoints))
    #         total = total[np.argsort(total[:, 1])]

    #         Temp = np.zeros_like(eachLengthPoints)
    #         Temp[-lostPoints:] = 1
    #         total[:, 1] += Temp
    #         total = total[np.argsort(total[:, 0])]

    #         eachLengthPoints = total[:, 1]
    #     elif eachLengthPoints.sum() > 100:
    #         lostPoints = eachLengthPoints.sum() - ResampleNum
    #         Temp = np.zeros_like(eachLengthPoints)
    #         Temp[0:lostPoints] = 1
    #         eachLengthPoints += Temp

    #     else:
    #         pass
    #     if eachLengthPoints.sum() != ResampleNum:
    #         raise ValueError("重采样点数不符")
    #     # eachLengthPoints中存放着每条边应当重采样的点的数目
    #     for i, point in enumerate(points):
    #         try:
    #             nextPoint = points[i + 1]
    #         except:
    #             nextPoint = points[0]
    #         sectionPoints = np.linspace(point, nextPoint, eachLengthPoints[i])
    #         ResamplePoints = np.append(ResamplePoints, sectionPoints, axis=0)

    #     return ResamplePoints

    # def cal_cp_coordinates(self, polygon, bs_degree, cp_num, resample_num):
    #     ResamplePoints = self.Resample(polygon, resample_num)
    #     ResamplePoints = np.vstack((ResamplePoints, ResamplePoints[0]))
    #     bs = BS_curve(cp_num, bs_degree)
    #     paras = bs.estimate_parameters(ResamplePoints)
    #     knots = bs.get_knots()
    #     if bs.check():
    #         cp = bs.approximation(ResamplePoints)
    #     CtrlPoints = cp[:-1]

    #     return CtrlPoints.flatten().astype(np.int32)

    # def cal_contour_points(self, bs_cp):
    #     bs_cp_int = bs_cp.reshape((-1, 2))  # .astype(np.int32)
    #     bs_cp_int = np.append(bs_cp_int, bs_cp_int[0]).reshape((-1, 2))
    #     bs = BS_curve(self.cp_num, 4, cp=bs_cp_int)
    #     uq = np.linspace(0, 1, 51)
    #     bs.set_paras(uq)
    #     knots = bs.get_knots()

    #     points = bs.bs(uq)
    #     points = np.array(points).astype(np.int32)
    #     points = points[:-1]
    #     return points

    # def generate_gt_targets(self, img, text_polys, ignore_polys):
    #     """Generate ground truth target on each level.

    #     Args:
    #         img_size (list[int]): Shape of input image.
    #         text_polys (list[list[ndarray]]): A list of ground truth polygons.
    #         ignore_polys (list[list[ndarray]]): A list of ignored polygons.
    #     Returns:
    #         level_maps (list(ndarray)): A list of ground target on each level.
    #     """
    #     # h, w = img_size
    #     gt_bboxes = np.zeros((len(text_polys), 4), dtype=np.float32)
    #     gt_bs_cp = np.zeros(
    #         (len(text_polys), self.cp_num * 2), dtype=np.float32)
    #     gt_lables = np.zeros(len(text_polys), dtype=np.float32)
    #     # gt_bboxes = np.array([])
    #     # gt_bs_cp = np.array([])
    #     for j, poly in enumerate(text_polys):
    #         assert len(poly) == 1
    #         text_instance = [[poly[0][i], poly[0][i + 1]]
    #                          for i in range(0, len(poly[0]), 2)]
    #         polygon = np.array(text_instance, dtype=np.int).reshape((1, -1, 2))
    #         polygon = self.normalize_polygon(polygon[0])
    #         box_x, box_y, box_w, box_h = cv2.boundingRect(polygon)
    #         single_bs_cp = self.cal_cp_coordinates(
    #             polygon, self.bs_degree, self.cp_num, self.resample_num)
    #         gt_bboxes[j] = np.array(
    #             [box_x, box_y, box_x + box_w, box_y + box_h])
    #         # single_bs_cp = self.normalize_polygon(single_bs_cp.reshape(-1, 2), flip_flag).flatten()
    #         gt_bs_cp[j] = np.array(single_bs_cp)

    #         # polygon_color = mmcv.color_val('green')
    #         # cp_color = mmcv.color_val('red')

    #         # reconstrucationPoints = self.cal_contour_points(single_bs_cp)
    #         # # cv2.polylines(img, [polygon.reshape(-1, 1, 2)], True, color=polygon_color, thickness=2)
    #         # cv2.polylines(img, [reconstrucationPoints.reshape(-1, 1, 2)], True, color=polygon_color, thickness=2)
    #         # # cv2.polylines(img, [single_bs_cp.reshape(-1, 1, 2)], True, color=cp_color, thickness=2)
    #         # cv2.polylines(img, [polygon.reshape(-1, 1, 2)], True, color=cp_color, thickness=2)
    #         # cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 0, 0), thickness=2)
    #         # mmcv.imwrite(img, 'img.jpg')
    #     return gt_bboxes, gt_bs_cp, gt_lables
    def resample_polygon(self, polygon, n=400):
        """Resample one polygon with n points on its boundary.

        Args:
            polygon (list[float]): The input polygon.
            n (int): The number of resampled points.
        Returns:
            resampled_polygon (list[float]): The resampled polygon.
        """
        length = []

        for i in range(len(polygon)):
            p1 = polygon[i]
            if i == len(polygon) - 1:
                p2 = polygon[0]
            else:
                p2 = polygon[i + 1]
            length.append(((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5)

        total_length = sum(length)
        n_on_each_line = (np.array(length) / (total_length + 1e-8)) * n
        n_on_each_line = n_on_each_line.astype(np.int32)
        new_polygon = []

        for i in range(len(polygon)):
            num = n_on_each_line[i]
            p1 = polygon[i]
            if i == len(polygon) - 1:
                p2 = polygon[0]
            else:
                p2 = polygon[i + 1]

            if num == 0:
                continue

            dxdy = (p2 - p1) / num
            for j in range(num):
                point = p1 + dxdy * j
                new_polygon.append(point)

        return np.array(new_polygon)

    def normalize_polygon(self, polygon):
        """Normalize one polygon so that its start point is at right most.

        Args:
            polygon (list[float]): The origin polygon.
        Returns:
            new_polygon (lost[float]): The polygon with start point at right.
        """
        temp_polygon = polygon - polygon.mean(axis=0)
        x = np.abs(temp_polygon[:, 0])
        y = temp_polygon[:, 1]
        index_x = np.argsort(x)
        index_y = np.argmin(y[index_x[:8]])
        index = index_x[index_y]
        new_polygon = np.concatenate([polygon[index:], polygon[:index]])
        return new_polygon

    def poly2fourier(self, polygon, fourier_degree):
        """Perform Fourier transformation to generate Fourier coefficients ck
        from polygon.

        Args:
            polygon (ndarray): An input polygon.
            fourier_degree (int): The maximum Fourier degree K.
        Returns:
            c (ndarray(complex)): Fourier coefficients.
        """
        points = polygon[:, 0] + polygon[:, 1] * 1j
        c_fft = fft(points) / len(points)
        c = np.hstack((c_fft[-fourier_degree:], c_fft[:fourier_degree + 1]))
        return c

    def clockwise(self, c, fourier_degree):
        """Make sure the polygon reconstructed from Fourier coefficients c in
        the clockwise direction.

        Args:
            polygon (list[float]): The origin polygon.
        Returns:
            new_polygon (lost[float]): The polygon in clockwise point order.
        """
        if np.abs(c[fourier_degree + 1]) > np.abs(c[fourier_degree - 1]):
            return c
        elif np.abs(c[fourier_degree + 1]) < np.abs(c[fourier_degree - 1]):
            return c[::-1]
        else:
            if np.abs(c[fourier_degree + 2]) > np.abs(c[fourier_degree - 2]):
                return c
            else:
                return c[::-1]

    def cal_fourier_signature(self, polygon, fourier_degree):
        """Calculate Fourier signature from input polygon.

        Args:
              polygon (ndarray): The input polygon.
              fourier_degree (int): The maximum Fourier degree K.
        Returns:
              fourier_signature (ndarray): An array shaped (2k+1, 2) containing
                  real part and image part of 2k+1 Fourier coefficients.
        """
        resampled_polygon = self.resample_polygon(polygon)
        resampled_polygon = self.normalize_polygon(resampled_polygon)

        fourier_coeff = self.poly2fourier(resampled_polygon, fourier_degree)
        fourier_coeff = self.clockwise(fourier_coeff, fourier_degree)

        real_part = np.real(fourier_coeff).reshape((-1, 1))
        image_part = np.imag(fourier_coeff).reshape((-1, 1))
        fourier_signature = np.hstack([real_part, image_part])

        return fourier_signature

    def generate_gt_targets(self,  text_polys, ignore_polys):
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
        k = self.fourier_degree
        gt_fourier_map = np.zeros(
            (len(text_polys), k * 4 + 2), dtype=np.float32)
        gt_lables = np.zeros(len(text_polys), dtype=np.float32)
        for j, poly in enumerate(text_polys):
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            polygon = np.array(text_instance, dtype=np.int).reshape((1, -1, 2))
            box_x, box_y, box_w, box_h = cv2.boundingRect(polygon)
            fourier_coeff = self.cal_fourier_signature(polygon[0], k)
            gt_bboxes[j] = np.array(
                [box_x, box_y, box_x + box_w, box_y + box_h])

            gt_fourier_map[j] = np.array(fourier_coeff.flatten())
        return gt_bboxes, gt_fourier_map, gt_lables

    def generate_targets(self, results):
        """Generate the ground truth targets for FCENet.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """

        assert isinstance(results, dict)
        # img = results['img']

        polygon_masks = results['gt_masks'].masks
        polygon_masks_ignore = results['gt_masks_ignore'].masks

        # h, w, _ = results['img_shape']

        gt_bboxes, gt_fourier, gt_lables = self.generate_gt_targets(
            polygon_masks, polygon_masks_ignore)

        mapping = {'gt_bboxes': gt_bboxes,
                   'gt_fourier': gt_fourier, 'gt_lables': gt_lables}
        for key, value in mapping.items():
            results[key] = value

        return results
