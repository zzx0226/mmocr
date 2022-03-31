'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2022-01-10 17:58:26
LastEditors: Zhangzixu
LastEditTime: 2022-03-10 20:27:14
'''
dataset_type = 'IcdarDataset'
data_root = '/home/atom/Research_STD/Datasets/mmocr/icdar2015'

train = dict(type=dataset_type, ann_file=f'{data_root}/instances_training.json', img_prefix=f'{data_root}/imgs', pipeline=None)

test = dict(type=dataset_type, ann_file=f'{data_root}/instances_test.json', img_prefix=f'{data_root}/imgs', pipeline=None)

train_list = [train]

test_list = [test]
