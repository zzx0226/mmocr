'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2022-03-10 20:35:58
LastEditors: Zhangzixu
LastEditTime: 2022-03-12 19:02:26
'''
dataset_type = 'IcdarDataset'
data_root_ctw1500 = '/home/atom/Research_STD/Datasets/mmocr/ctw1500'
data_root_TT = '/home/atom/Research_STD/Datasets/mmocr/totaltext'

train = dict(type=dataset_type,
             ann_file=[f'{data_root_ctw1500}/instances_training.json', f'{data_root_TT}/instances_training.json'],
             img_prefix=[f'{data_root_ctw1500}/imgs', f'{data_root_TT}/imgs'],
             pipeline=None)

test = dict(type=dataset_type,
            ann_file=f'{data_root_ctw1500}/instances_test.json',
            img_prefix=f'{data_root_ctw1500}/imgs',
            pipeline=None)

train_list = [train]

test_list = [test]
