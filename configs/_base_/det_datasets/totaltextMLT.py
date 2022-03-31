'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2022-03-12 14:46:06
LastEditors: Zhangzixu
LastEditTime: 2022-03-12 14:47:44
'''
dataset_type = 'IcdarDataset'
data_root_TT = '/home/atom/Research_STD/Datasets/mmocr/totaltext'
data_root_MLT = '/home/atom/Research_STD/Datasets/mmocr/icdar2017'

train = dict(type=dataset_type,
             ann_file=[f'{data_root_TT}/instances_training.json', f'{data_root_MLT}/instances_training.json'],
             img_prefix=[f'{data_root_TT}/imgs', f'{data_root_MLT}'],
             pipeline=None)

test = dict(type=dataset_type, ann_file=f'{data_root_TT}/instances_test.json', img_prefix=f'{data_root_TT}/imgs', pipeline=None)

train_list = [train]

test_list = [test]