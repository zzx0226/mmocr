# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
from shapely.geometry.polygon import LinearRing
from B_Spline import BS_curve
import torch
import json
from tqdm import tqdm
import os


class generateCP():
    def __init__(self, cp_num=8, bs_degree=4, ResampleNum=100):
        self.cp_num = cp_num
        self.bs_degree = bs_degree
        self.ResampleNum = ResampleNum

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
            print("Points num >100")

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
        # ResamplePoints = ResamplePoints.astype(np.int32)

        return ResamplePoints

    def __call__(self, results):

        # cp_list = np.zeros([len(results['ann_info']['masks']), self.cp_num, 2])
        # cp_list = []
        #'segmentation'
        for i, gt_contour in enumerate(results['segmentation']):
            Points = self.Resample(np.array(gt_contour).reshape(-1, 2), self.ResampleNum)
            xx = []
            yy = []
            bs = BS_curve(self.cp_num, self.bs_degree)
            xx = Points[:, 0]
            yy = Points[:, 1]
            data = np.array([xx, yy]).T
            paras = bs.estimate_parameters(data)
            knots = bs.get_knots()
            if bs.check():
                cp = bs.approximation(data)
            uq = np.linspace(0, 1, self.ResampleNum + 1)
            y = bs.bs(uq)
            CtrlPoints = cp[:-1]
            # cp_list.append(torch.tensor(CtrlPoints.flatten()))
            # cp_list.append(list(CtrlPoints.flatten()))

            results['bs_cp'] = list(CtrlPoints.flatten().astype(np.int32).astype(float))
            # print('1')


cp_num = 14
bs_degree = 4
# data = json.loads("../data/CTW1500/JPGImages/train_labels.json")
train_json = "/home/atom/Research_STD/Datasets/mmocr/ctw1500/instances_training.json"
test_json = "/home/atom/Research_STD/Datasets/mmocr/ctw1500/instances_test.json"

train_json_with_cp = f"/home/atom/Research_STD/Datasets/mmocr/ctw1500/train_labels_with_bs_{bs_degree}_cp_{cp_num}.json"
test_json_with_cp = f"/home/atom/Research_STD/Datasets/mmocr/ctw1500/CTW_test_with_bs_{bs_degree}_cp_{cp_num}.json"

# train_json_with_bc = "data/CTW1500/JPGImages/train_ctw1500_maxlen100_v2.json"
generate_CP = generateCP(cp_num=cp_num, bs_degree=bs_degree, ResampleNum=100)

# with open(train_json, "r") as f:
#     lable = json.load(f)
#     print("Load annotations")
#     annotations = lable['annotations']
#     for annotation in tqdm(annotations):
#         generate_CP(annotation)

# with open(train_json_with_cp, "w") as f:
#     json.dump(lable, f)

# with open(test_json, "r") as f:
#     lable = json.load(f)
#     print("Load annotations")
#     annotations = lable['annotations']
#     for annotation in tqdm(annotations):
#         generate_CP(annotation)

# with open(test_json_with_cp, "w") as f:
#     json.dump(lable, f)

with open(train_json_with_cp, "r") as f:
    lable = json.load(f)
    print("Load annotations with cp")
# with open(train_json_with_cp, "r") as f1:
#     bc = json.load(f1)
#     # print("Load annotations with bc")
#     annotations = bc['annotations']
#     for annotation in tqdm(annotations):
#         annotation['bs_cp'] = annotation['bezier_pts']

# with open(train_json_with_bc, "w") as f:
#     json.dump(bc, f)
