'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2021-12-21 21:42:50
LastEditors: Zhangzixu
LastEditTime: 2022-03-13 09:00:29
'''
_base_ = [
    '../../_base_/runtime_1e.py', '../../_base_/schedules/schedule_adam_step_150e.py',
    '../../_base_/det_models/wlnet_r50dcnv2_fpn_test.py', '../../_base_/det_datasets/icdar2015MLT.py',
    '../../_base_/det_pipelines/wlnet_pipeline.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_icdar2015 = {{_base_.train_pipeline_icdar2015}}
test_pipeline_icdar2015 = {{_base_.test_pipeline_icdar2015}}

data = dict(samples_per_gpu=7,
            workers_per_gpu=7,
            val_dataloader=dict(samples_per_gpu=1),
            test_dataloader=dict(samples_per_gpu=1),
            train=dict(type='UniformConcatDataset', datasets=train_list, pipeline=train_pipeline_icdar2015),
            val=dict(type='UniformConcatDataset', datasets=test_list, pipeline=test_pipeline_icdar2015),
            test=dict(type='UniformConcatDataset', datasets=test_list, pipeline=test_pipeline_icdar2015))

evaluation = dict(interval=1, metric='hmean-iou')
#