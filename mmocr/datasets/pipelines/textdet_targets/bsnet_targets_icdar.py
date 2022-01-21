# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES
from numpy.fft import fft
from numpy.linalg import norm

import mmocr.utils.check_argument as check_argument
from .textsnake_targets import TextSnakeTargets
from geomdl import fitting
from shapely.geometry.polygon import LinearRing
from geomdl import BSpline
from geomdl import knotvector


@PIPELINES.register_module()
class BSNetTargets_icdar(TextSnakeTargets):
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

    def __init__(self,
                 resample_step=4.0,
                 bs_degree=4,
                 cp_num=14,
                 resample_num=10,
                 center_region_shrink_ratio=0.3,
                 level_size_divisors=(8, 16, 32),
                 level_proportion_range=((0, 0.4), (0.3, 0.7), (0.6, 1.0))):

        super().__init__()
        assert isinstance(level_size_divisors, tuple)
        assert isinstance(level_proportion_range, tuple)
        assert len(level_size_divisors) == len(level_proportion_range)
        self.resample_step = resample_step
        self.center_region_shrink_ratio = center_region_shrink_ratio
        self.level_size_divisors = level_size_divisors
        self.level_proportion_range = level_proportion_range

        self.bs_degree = bs_degree
        self.cp_num = cp_num
        self.resample_num = resample_num
        self.sample_size = 40

    def normalize_bs_polygon(self, polygon):
        pRing = LinearRing(polygon)
        if not pRing.is_ccw:
            polygon = np.flipud(polygon)

        index = np.argsort(np.sqrt(polygon[:, 0]**2 + polygon[:, 1]**2) + polygon[:, 0])[0]
        new_polygon = np.concatenate([polygon[index:], polygon[:index]])
        return new_polygon

    def Resample(self, points, ResampleNum):
        ResamplePoints = np.empty([0, 2])
        eachLength = []
        for i, point in enumerate(points):
            try:
                nextPoint = points[i + 1]
            except:
                nextPoint = points[0]
            eachLength.append(int(np.linalg.norm((point - nextPoint))))
        eachLength = np.array(eachLength)
        max_length = np.argmax(eachLength)
        addTwoPoints = []
        if max_length == 0:
            addTwoPoints = [True, False, True, False]
        elif max_length == 2:
            addTwoPoints = [True, False, True, False]
        elif max_length == 1:
            addTwoPoints = [False, True, False, True]
        elif max_length == 3:
            addTwoPoints = [False, True, False, True]
        # ResamplePoints = np.array([])
        for i, add in enumerate(addTwoPoints):
            try:
                nextPoint = points[i + 1]
            except:
                nextPoint = points[0]
            if add == True:
                sectionPoints = np.linspace(points[i], nextPoint, 4)[:-1]
                ResamplePoints = np.append(ResamplePoints, sectionPoints, axis=0)
            else:
                sectionPoints = np.linspace(points[i], nextPoint, 3)[:-1]
                ResamplePoints = np.append(ResamplePoints, sectionPoints, axis=0)
        return ResamplePoints

    def generate_center_region_mask(self, img_size, text_polys):
        """Generate text center region mask.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            center_region_mask (ndarray): The text center region mask.
        """

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size

        center_region_mask = np.zeros((h, w), np.uint8)

        center_region_boxes = []
        for poly in text_polys:
            assert len(poly) == 1
            polygon_points = poly[0].reshape(-1, 2)
            _, _, top_line, bot_line = self.reorder_poly_edge(polygon_points)
            resampled_top_line, resampled_bot_line = self.resample_sidelines(top_line, bot_line, self.resample_step)
            resampled_bot_line = resampled_bot_line[::-1]
            center_line = (resampled_top_line + resampled_bot_line) / 2

            line_head_shrink_len = norm(resampled_top_line[0] - resampled_bot_line[0]) / 4.0
            line_tail_shrink_len = norm(resampled_top_line[-1] - resampled_bot_line[-1]) / 4.0
            head_shrink_num = int(line_head_shrink_len // self.resample_step)
            tail_shrink_num = int(line_tail_shrink_len // self.resample_step)
            if len(center_line) > head_shrink_num + tail_shrink_num + 2:
                center_line = center_line[head_shrink_num:len(center_line) - tail_shrink_num]
                resampled_top_line = resampled_top_line[head_shrink_num:len(resampled_top_line) - tail_shrink_num]
                resampled_bot_line = resampled_bot_line[head_shrink_num:len(resampled_bot_line) - tail_shrink_num]

            for i in range(0, len(center_line) - 1):
                tl = center_line[i] + (resampled_top_line[i] - center_line[i]) * self.center_region_shrink_ratio
                tr = center_line[i + 1] + (resampled_top_line[i + 1] - center_line[i + 1]) * self.center_region_shrink_ratio
                br = center_line[i + 1] + (resampled_bot_line[i + 1] - center_line[i + 1]) * self.center_region_shrink_ratio
                bl = center_line[i] + (resampled_bot_line[i] - center_line[i]) * self.center_region_shrink_ratio
                current_center_box = np.vstack([tl, tr, br, bl]).astype(np.int32)
                center_region_boxes.append(current_center_box)

        cv2.fillPoly(center_region_mask, center_region_boxes, 1)
        return center_region_mask

    def cal_cp_coordinates(self, polygon, bs_degree, resample_num):
        ResamplePoints = self.Resample(polygon, resample_num)
        CtrlPoints = fitting.approximate_curve(ResamplePoints.tolist(), bs_degree).ctrlpts
        CtrlPoints = np.array(CtrlPoints).reshape(-1, 2)

        crv = BSpline.Curve()
        crv.degree = self.bs_degree
        crv.ctrlpts = CtrlPoints.tolist()
        crv.knotvector = knotvector.generate(crv.degree, crv.ctrlpts_size)
        crv.sample_size = self.sample_size
        points_top = np.array(crv.evalpts)
        return CtrlPoints

    def generate_cp_maps(self, img_size, text_polys):

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size
        cp_num = self.cp_num
        x_map = np.zeros((cp_num, h, w), dtype=np.float32)
        y_map = np.zeros((cp_num, h, w), dtype=np.float32)

        for poly in text_polys:
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]] for i in range(0, len(poly[0]), 2)]
            mask = np.zeros((h, w), dtype=np.uint8)
            polygon = np.array(text_instance).reshape((1, -1, 2))

            polygon = self.normalize_bs_polygon(polygon[0])
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
            cp_coordinates = self.cal_cp_coordinates(polygon, self.bs_degree, self.resample_num)
            for i in range(0, cp_num):
                x_map[i, :, :] = mask * cp_coordinates[i, 0] + \
                    (1 - mask) * x_map[i, :, :]
                y_map[i, :, :] = mask * cp_coordinates[i, 1] + \
                    (1 - mask) * y_map[i, :, :]

        return x_map, y_map

    def generate_level_targets(self, img_size, text_polys, ignore_polys):
        """Generate ground truth target on each level.

        Args:
            img_size (list[int]): Shape of input image.
            text_polys (list[list[ndarray]]): A list of ground truth polygons.
            ignore_polys (list[list[ndarray]]): A list of ignored polygons.
        Returns:
            level_maps (list(ndarray)): A list of ground target on each level.
        """
        h, w = img_size
        lv_size_divs = self.level_size_divisors
        lv_proportion_range = self.level_proportion_range
        lv_text_polys = [[] for i in range(len(lv_size_divs))]
        lv_ignore_polys = [[] for i in range(len(lv_size_divs))]
        level_maps = []
        for poly in text_polys:
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]] for i in range(0, len(poly[0]), 2)]
            polygon = np.array(text_instance, dtype=np.int).reshape((1, -1, 2))
            _, _, box_w, box_h = cv2.boundingRect(polygon)
            proportion = max(box_h, box_w) / (h + 1e-8)

            for ind, proportion_range in enumerate(lv_proportion_range):
                if proportion_range[0] < proportion < proportion_range[1]:
                    lv_text_polys[ind].append([poly[0] / lv_size_divs[ind]])

        for ignore_poly in ignore_polys:
            assert len(ignore_poly) == 1
            text_instance = [[ignore_poly[0][i], ignore_poly[0][i + 1]] for i in range(0, len(ignore_poly[0]), 2)]
            polygon = np.array(text_instance, dtype=np.int).reshape((1, -1, 2))
            _, _, box_w, box_h = cv2.boundingRect(polygon)
            proportion = max(box_h, box_w) / (h + 1e-8)

            for ind, proportion_range in enumerate(lv_proportion_range):
                if proportion_range[0] < proportion < proportion_range[1]:
                    lv_ignore_polys[ind].append([ignore_poly[0] / lv_size_divs[ind]])

        for ind, size_divisor in enumerate(lv_size_divs):
            current_level_maps = []
            level_img_size = (h // size_divisor, w // size_divisor)

            text_region = self.generate_text_region_mask(level_img_size, lv_text_polys[ind])[None]
            current_level_maps.append(text_region)

            center_region = self.generate_center_region_mask(level_img_size, lv_text_polys[ind])[None]
            current_level_maps.append(center_region)

            effective_mask = self.generate_effective_mask(level_img_size, lv_ignore_polys[ind])[None]
            current_level_maps.append(effective_mask)

            cp_x_maps, cp_y_maps = self.generate_cp_maps(level_img_size, lv_text_polys[ind])
            current_level_maps.append(cp_x_maps)
            current_level_maps.append(cp_y_maps)

            level_maps.append(np.concatenate(current_level_maps))

        return level_maps

    def generate_targets(self, results):
        """Generate the ground truth targets for FCENet.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """

        assert isinstance(results, dict)

        polygon_masks = results['gt_masks'].masks
        polygon_masks_ignore = results['gt_masks_ignore'].masks

        self.flip_flag1 = results['Stage1_flip']
        self.flip_flag2 = results['flip']

        h, w, _ = results['img_shape']

        level_maps = self.generate_level_targets((h, w), polygon_masks, polygon_masks_ignore)

        results['mask_fields'].clear()  # rm gt_masks encoded by polygons
        mapping = {'p3_maps': level_maps[0], 'p4_maps': level_maps[1], 'p5_maps': level_maps[2]}
        for key, value in mapping.items():
            results[key] = value

        return results
