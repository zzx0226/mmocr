# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np

from mmocr.models.builder import POSTPROCESSOR
from .base_postprocessor import BasePostprocessor
from .utils import fill_hole, fourier2poly, poly_nms
from .B_Spline import BS_curve

import multiprocessing
from functools import partial


@POSTPROCESSOR.register_module()
class BSPostprocessor(BasePostprocessor):
    """Decoding predictions of FCENet to instances.

    Args:
        fourier_degree (int): The maximum Fourier transform degree k.
        num_reconstr_points (int): The points number of the polygon
            reconstructed from predicted Fourier coefficients.
        text_repr_type (str): Boundary encoding type 'poly' or 'quad'.
        scale (int): The down-sample scale of the prediction.
        alpha (float): The parameter to calculate final scores. Score_{final}
                = (Score_{text region} ^ alpha)
                * (Score_{text center region}^ beta)
        beta (float): The parameter to calculate final score.
        score_thr (float): The threshold used to filter out the final
            candidates.
        nms_thr (float): The threshold of nms.
    """

    def __init__(
            self,
            #  fourier_degree,
            bs_degree,
            cp_num,
            num_reconstr_points,
            # text_repr_type='poly',
            alpha=1.0,
            beta=2.0,
            score_thr=0.3,
            nms_thr=0.1,
            **kwargs):
        # super().__init__(text_repr_type)
        # self.fourier_degree = fourier_degree
        self.bs_degree = bs_degree
        self.cp_num = cp_num

        self.num_reconstr_points = num_reconstr_points
        self.alpha = alpha
        self.beta = beta
        self.score_thr = score_thr
        self.nms_thr = nms_thr

    def init_pool(self, array):
        global glob_array
        glob_array = array

    def Cal_Contour(self, i, eachContours):

        # for i, eachContour in enumerate(eachContours):
        # global AllContours
        eachContours = eachContours[i]
        eachContours = np.vstack((eachContours, eachContours[0]))
        bs = BS_curve(self.cp_num, self.bs_degree, cp=eachContours)
        uq = np.linspace(0, 1, self.num_reconstr_points + 1)
        bs.set_paras(uq)
        knots = bs.get_knots()
        points = bs.bs(uq)
        points = points[:-1]
        glob_array[i * self.num_reconstr_points *
                   2:(i + 1) * self.num_reconstr_points * 2] = np.array(points).flatten()

    def __call__(self, preds, scale):
        """
        Args:
            preds (list[Tensor]): Classification prediction and regression
                prediction.
            scale (float): Scale of current layer.

        Returns:
            list[list[float]]: The instance boundary and confidence.
        """
        assert isinstance(preds, list)
        assert len(preds) == 2

        cls_pred = preds[0][0]
        tr_pred = cls_pred[0:2].softmax(dim=0).data.cpu().numpy()
        tcl_pred = cls_pred[2:].softmax(dim=0).data.cpu().numpy()

        reg_pred = preds[1][0].permute(1, 2, 0).data.cpu().numpy()
        x_pred = reg_pred[:, :, :self.cp_num]
        y_pred = reg_pred[:, :, self.cp_num:]

        score_pred = (tr_pred[1]**self.alpha) * (tcl_pred[1]**self.beta)
        tr_pred_mask = (score_pred) > self.score_thr
        tr_mask = fill_hole(tr_pred_mask)

        tr_contours, _ = cv2.findContours(tr_mask.astype(
            np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # opencv4

        mask = np.zeros_like(tr_mask)
        boundaries = []
        for cont in tr_contours:
            deal_map = mask.copy().astype(np.int8)
            cv2.drawContours(deal_map, [cont], -1, 1, -1)

            score_map = score_pred * deal_map
            score_mask = score_map > 0
            xy_text = np.argwhere(score_mask)
            # xy_text = np.hstack((xy_text[:, 1], xy_text[:, 0]))

            xs, ys = x_pred[score_mask], y_pred[score_mask]

            # dx = xy_text[:, 0] / scale
            # dy = xy_text[:, 1] / scale
            # xs = xs - dx.repeat(8).reshape(len(xs), -1)
            # ys = ys - dy.repeat(8).reshape(len(xs), -1)
            # Contours = xs.copy()
            # for i in range(xs.shape[1]):
            #     Contours = np.insert(Contours, i * 2 + 1, values=ys[:, i], axis=1)
            # Contours = Contours.reshape(len(Contours), -1, 2) * scale

            # AllContours = np.zeros([int(len(Contours)), self.num_reconstr_points, 2])
            # array_shared = multiprocessing.RawArray('f', AllContours.ravel())
            # func_partial = partial(self.Cal_Contour, eachContours=Contours)
            # p = multiprocessing.Pool(initializer=self.init_pool, initargs=(array_shared, ))
            # p.map(func_partial, range(Contours.shape[0]))
            # p.close()
            # p.join()
            # AllContours = np.frombuffer(array_shared, np.float32).reshape(int(len(Contours)), -1)
            # score = score_map[score_mask].reshape(-1, 1)
            # polygons = np.column_stack((AllContours, score[:, 0])).tolist()
            polygons = []
            score = score_map[score_mask].flatten()

            index = np.argmax(score)
            c = np.vstack((xs[index], ys[index])).T
            c *= scale
            bs_cp = np.append(c, c[0]).reshape((-1, 2))
            bs = BS_curve(self.cp_num, self.bs_degree, cp=bs_cp)
            uq = np.linspace(0, 1, self.num_reconstr_points + 1)
            bs.set_paras(uq)
            knots = bs.get_knots()

            points = bs.bs(uq)
            points = points[:-1]
            # split_index = int(self.cp_num / 2)
            # eachContours = c.reshape((-1, 2))
            # TempContours1 = np.vstack(
            #     (eachContours[split_index:], eachContours[:split_index]))
            # TempContours1 = np.vstack((TempContours1, TempContours1[0]))
            # bs = BS_curve(self.cp_num, self.bs_degree, cp=TempContours1)
            # uq = np.linspace(0, 1, self.num_reconstr_points + 1)
            # bs.set_paras(uq)
            # knots = bs.get_knots()
            # points1 = bs.bs(uq)
            # points1 = points1[:-1]
            # points1 = points1[25:75]

            # TempContours2 = np.vstack((eachContours, eachContours[0]))
            # bs = BS_curve(self.cp_num, self.bs_degree, cp=TempContours2)
            # uq = np.linspace(0, 1, self.num_reconstr_points + 1)
            # bs.set_paras(uq)
            # knots = bs.get_knots()
            # points2 = bs.bs(uq)
            # points2 = points2[:-1]
            # points2 = points2[25:75]

            # points = np.append(points1, points2)
            points = np.append(points.flatten(), score[index]).tolist()

            polygons.append(points)
            # for i, (x, y) in enumerate(zip(xs, ys)):
            #     c = np.vstack((x, y)).T
            #     # c = x + y * 1j
            #     # c[:, self.fourier_degree] = c[:, self.fourier_degree] + dxy
            #     # c = c * scale+xy_text[i]
            #     # c = (c+xy_text[i]) * scale
            #     c *= scale

            #     bs_cp = np.append(c, c[0]).reshape((-1, 2))
            #     bs = BS_curve(self.cp_num, self.bs_degree, cp=bs_cp)
            #     uq = np.linspace(0, 1, self.num_reconstr_points + 1)
            #     bs.set_paras(uq)
            #     knots = bs.get_knots()

            #     points = bs.bs(uq)
            #     points = points[:-1]

            #     points = np.append(points.flatten(), score[i]).tolist()
            #     polygons.append(points)
            # polygons = poly_nms(polygons, self.nms_thr)
            boundaries = boundaries + polygons

        # boundaries = poly_nms(boundaries, self.nms_thr)

        # if self.text_repr_type == 'quad':
        #     new_boundaries = []
        #     for boundary in boundaries:
        #         poly = np.array(boundary[:-1]).reshape(-1, 2).astype(np.float32)
        #         score = boundary[-1]
        #         points = cv2.boxPoints(cv2.minAreaRect(poly))
        #         points = np.int0(points)
        #         new_boundaries.append(points.reshape(-1).tolist() + [score])

        return boundaries
