'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2022-01-04 13:28:06
LastEditors: Zhangzixu
LastEditTime: 2022-01-04 16:47:04
'''
_base_ = [
    '../../_base_/runtime_10e.py', '../../_base_/schedules/schedule_sgd_1500e.py',
    '../../_base_/det_models/fcenet_fcos_r50dcnv2_fpn.py', '../../_base_/det_datasets/ctw1500.py',
    '../../_base_/det_pipelines/fcenet_fcos_pipeline.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_ctw1500 = {{_base_.train_pipeline_ctw1500}}
test_pipeline_ctw1500 = {{_base_.test_pipeline_ctw1500}}

data = dict(samples_per_gpu=4,
            workers_per_gpu=4,
            val_dataloader=dict(samples_per_gpu=1),
            test_dataloader=dict(samples_per_gpu=1),
            train=dict(type='UniformConcatDataset',
                       datasets=train_list, pipeline=train_pipeline_ctw1500),
            val=dict(type='UniformConcatDataset', datasets=test_list,
                     pipeline=test_pipeline_ctw1500),
            test=dict(type='UniformConcatDataset', datasets=test_list, pipeline=test_pipeline_ctw1500))

evaluation = dict(interval=1500, metric='hmean-iou')
