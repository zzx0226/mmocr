# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch
import json
from tqdm import tqdm
import os
import pywt

wavelet_type = 'haar'


class generateWL():
    def __init__(self, wavelet_type='haar'):
        self.wavelet_type = wavelet_type

    def __call__(self, results):

        for gt_contour in results['segmentation']:
            Points = np.array(gt_contour).reshape(-1, 2)
            cA, (cH, cV, cD) = pywt.wavedec2(Points, 'haar')
            wavelet_coeffs = np.array([cA, cH, cV, cD]).flatten().astype(float)

            results[f'wl_{wavelet_type}'] = list(wavelet_coeffs)


data_root = '/home/atom/Research_STD/Datasets/mmocr/ctw1500'

train_json = "/home/atom/Research_STD/Datasets/mmocr/ctw1500/instances_training.json"
test_json = "/home/atom/Research_STD/Datasets/mmocr/ctw1500/instances_test.json"

train_json_with_wl = f"/home/atom/Research_STD/Datasets/mmocr/ctw1500/instances_training_with_{wavelet_type}.json"
test_json_with_wl = f"/home/atom/Research_STD/Datasets/mmocr/ctw1500/instances_test_with_{wavelet_type}.json"

# train_json_with_bc = "data/CTW1500/JPGImages/train_ctw1500_maxlen100_v2.json"
generate_WL = generateWL(wavelet_type=wavelet_type)

with open(train_json, "r") as f:
    lable = json.load(f)
    print("Load annotations")
    annotations = lable['annotations']
    for annotation in tqdm(annotations):
        generate_WL(annotation)

with open(train_json_with_wl, "w") as f:
    json.dump(lable, f)

with open(test_json, "r") as f:
    lable = json.load(f)
    print("Load annotations")
    annotations = lable['annotations']
    for annotation in tqdm(annotations):
        generate_WL(annotation)

with open(test_json_with_wl, "w") as f:
    json.dump(lable, f)
