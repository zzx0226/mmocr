# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np

from mmocr.models.builder import POSTPROCESSOR
from .base_postprocessor import BasePostprocessor
from .utils import fill_hole, fourier2poly, poly_nms
import pywt


@POSTPROCESSOR.register_module()
class WLPostprocessor(BasePostprocessor):
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

    def __init__(self, num_reconstr_points, wavelet_type='sym5', alpha=1.0, beta=2.0, score_thr=0.3, nms_thr=0.1, **kwargs):
        # super().__init__(text_repr_type)
        self.wavelet_type = wavelet_type

        self.num_reconstr_points = num_reconstr_points
        self.alpha = alpha
        self.beta = beta
        self.score_thr = score_thr
        self.nms_thr = nms_thr

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
        real_preds = reg_pred[:, :, :20]
        imag_preds = reg_pred[:, :, 21:-1]

        centers_x = reg_pred[:, :, 20]
        centers_y = reg_pred[:, :, -1]

        score_pred = (tr_pred[1]**self.alpha) * (tcl_pred[1]**self.beta)
        tr_pred_mask = (score_pred) > self.score_thr
        tr_mask = fill_hole(tr_pred_mask)

        tr_contours, _ = cv2.findContours(tr_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # opencv4

        mask = np.zeros_like(tr_mask)
        boundaries = []
        for cont in tr_contours:
            deal_map = mask.copy().astype(np.int8)
            cv2.drawContours(deal_map, [cont], -1, 1, -1)

            score_map = score_pred * deal_map
            score_mask = score_map > 0
            xy_texts = np.argwhere(score_mask)
            dxy = xy_texts[:, 1] + xy_texts[:, 0] * 1j
            dxy_c = dxy * scale

            real_pred, imag_pred = real_preds[score_mask], imag_preds[score_mask]
            center_x, center_y = centers_x[score_mask], centers_y[score_mask]
            score = score_map[score_mask].reshape(-1, 1)
            polygons = []
            for i, (real, imag, dxy, x, y) in enumerate(zip(real_pred, imag_pred, dxy_c, center_x, center_y)):
                # c = np.vstack((real, imag)).T
                c = real + imag * 1j
                # c[:, self.fourier_degree] = c[:, self.fourier_degree] + dxy
                wl_coeff = c * scale
                wl_coeff = [wl_coeff, np.zeros(20), np.zeros(31), np.zeros(54)]

                points = pywt.waverec(wl_coeff, self.wavelet_type)
                points = np.append(points.real, points.imag).reshape(2, -1).T
                points[:, 0] = points[:, 0] + dxy.real + x * scale
                points[:, 1] = points[:, 1] + dxy.imag + y * scale

                points = np.append(points.flatten(), score[i][0]).tolist()
                polygons.append(points)
            polygons = poly_nms(polygons, self.nms_thr)
            boundaries = boundaries + polygons

        boundaries = poly_nms(boundaries, self.nms_thr)

        # if self.text_repr_type == 'quad':
        #     new_boundaries = []
        #     for boundary in boundaries:
        #         poly = np.array(boundary[:-1]).reshape(-1, 2).astype(np.float32)
        #         score = boundary[-1]
        #         points = cv2.boxPoints(cv2.minAreaRect(poly))
        #         points = np.int0(points)
        #         new_boundaries.append(points.reshape(-1).tolist() + [score])

        return boundaries
