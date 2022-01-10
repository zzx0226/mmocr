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
class BSNetFcosTargets_tb(TextSnakeTargets):
    def __init__(self, bs=4, cp=5):

        super().__init__()

        self.bs_degree = bs
        self.cp_num = cp

    def _normalize_polygon(self, polygon):
        """Normalize one polygon so that its start point is at right most.

        Args:
            polygon (list[float]): The origin polygon.
        Returns:
            new_polygon (lost[float]): The polygon with start point at right.
        """
        num = len(polygon)
        temp_polygon = polygon - polygon.mean(axis=0)
        x = np.abs(temp_polygon[:, 0])
        y = temp_polygon[:, 1]
        index_x = np.argsort(x)
        index_y = np.argmin(y[index_x[:8]])
        index = index_x[index_y]
        num = num - index
        new_polygon = np.concatenate([polygon[index:], polygon[:index]])
        return new_polygon

    def normalize_polygon(self, polygon):
        """Normalize one polygon so that its start point is at right most.

        Args:
            polygon (list[float]): The origin polygon.
        Returns:
            new_polygon (lost[float]): The polygon with start point at right.
        """
        # if self.flip_flag2:
        #     polygon = np.flipud(polygon)
        # polygon = self._normalize_polygon(polygon)
        # return polygon

        # index = np.argsort(polygon[:, 0])[0]
        # anno_len = len(polygon)
        # if (self.flip1_type == 'N' and self.flip2_type == True):
        #     polygon = np.flipud(polygon)
        #     index = 7  # polygon = np.flipud(polygon)
        # elif (self.flip1_type == 'N' and self.flip2_type == False):
        #     index = 0
        # elif (self.flip1_type == 'H' and self.flip2_type == True):
        #     polygon = np.flipud(polygon)
        #     index = 7  # polygon = np.flipud(polygon)
        # elif (self.flip1_type == 'H' and self.flip2_type == False):
        #     index = 0
        # elif (self.flip1_type == 'V' and self.flip2_type == True):
        #     polygon = np.flipud(polygon)
        #     index = 7  # polygon = np.flipud(polygon)
        # elif (self.flip1_type == 'V' and self.flip2_type == False):
        #     index = 0
        # elif (self.flip1_type == 'B' and self.flip2_type == True):
        #     polygon = np.flipud(polygon)
        #     index = 7
        # elif (self.flip1_type == 'B' and self.flip2_type == False):
        #     index = 7
        # else:
        #     index = 0
        # new_polygon = np.concatenate([polygon[index:], polygon[:index]])
        if ((self.flip1_type == 'V' or self.flip1_type == 'H') and self.flip2_type == False)\
                or ((self.flip1_type == 'N' or self.flip1_type == 'B') and self.flip2_type == True):
            polygon = np.flipud(polygon)
        box_x, box_y, box_w, box_h = cv2.boundingRect(polygon)
        if box_w > box_h:
            index = np.argsort(np.sqrt(polygon[:, 0]**2 + polygon[:, 1]**2) + polygon[:, 0])[0]
        else:
            left_x = box_x
            left_y = box_y + box_h
            index = np.argsort(np.sqrt((polygon[:, 0] - left_x)**2 + (polygon[:, 1] - left_y)**2) + (polygon[:, 0] - left_x))[0]
        new_polygon = np.concatenate([polygon[index:], polygon[:index]])
        return new_polygon

    def cal_cp_coordinates(self, polygon, bs_degree, cp_num):
        SplitIndex = int(len(polygon) / 2)
        TopLine = polygon[:SplitIndex]
        BottomLine = polygon[SplitIndex:]
        # BottomLine = BottomLine[::-1]

        bs = BS_curve(cp_num - 1, bs_degree)
        paras = bs.estimate_parameters(TopLine)
        knots = bs.get_knots()
        if bs.check():
            cp = bs.approximation(TopLine)
        TopCtrlPoints = cp

        bs = BS_curve(cp_num - 1, bs_degree)
        paras = bs.estimate_parameters(BottomLine)
        knots = bs.get_knots()
        if bs.check():
            cp = bs.approximation(BottomLine)
        BottomCtrlPoints = cp

        CtrlPoints = np.append(TopCtrlPoints, BottomCtrlPoints)

        return CtrlPoints.flatten()

    def cal_contour_points(self, bs_cp):
        bs_cp_int = bs_cp.reshape((-1, 2))  # .astype(np.int32)
        bs_cp_int = np.append(bs_cp_int, bs_cp_int[0]).reshape((-1, 2))
        bs = BS_curve(self.cp_num, 4, cp=bs_cp_int)
        uq = np.linspace(0, 1, 51)
        bs.set_paras(uq)
        knots = bs.get_knots()

        points = bs.bs(uq)
        points = np.array(points).astype(np.int32)
        points = points[:-1]
        return points

    def generate_gt_targets(self, text_polys, ignore_polys):
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
        gt_bs_cp = np.zeros((len(text_polys), self.cp_num * 4), dtype=np.float32)
        gt_lables = np.zeros(len(text_polys), dtype=np.float32)
        # gt_bboxes = np.array([])
        # gt_bs_cp = np.array([])
        for j, poly in enumerate(text_polys):
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]] for i in range(0, len(poly[0]), 2)]
            polygon = np.array(text_instance, dtype=np.int).reshape((-1, 2))
            polygon = self.normalize_polygon(polygon)
            box_x, box_y, box_w, box_h = cv2.boundingRect(polygon)
            single_bs_cp = self.cal_cp_coordinates(polygon, self.bs_degree, self.cp_num)
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
        # img = results['img']
        self.flip_flag1 = results['Stage1_flip']
        self.flip_flag2 = results['flip']

        self.flip1_type = results['Stage1_Type']
        self.flip2_type = results['flip']

        polygon_masks = results['gt_masks'].masks
        polygon_masks_ignore = results['gt_masks_ignore'].masks

        # h, w, _ = results['img_shape'] img,

        gt_bboxes, gt_bs_cp, gt_lables = self.generate_gt_targets(polygon_masks, polygon_masks_ignore)

        mapping = {'gt_bboxes': gt_bboxes, 'gt_cp': gt_bs_cp, 'gt_lables': gt_lables}
        for key, value in mapping.items():
            results[key] = value

        return results
