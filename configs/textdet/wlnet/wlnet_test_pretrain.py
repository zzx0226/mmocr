'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2022-01-27 21:50:16
LastEditors: Zhangzixu
LastEditTime: 2022-03-05 14:38:48
'''
_base_ = [
    '../../_base_/runtime_10e.py', '../../_base_/schedules/schedule_adam_step_1500e_bsnet_tb.py',
    '../../_base_/det_models/wlnet_r50dcnv2_fpn.py', '../../_base_/det_datasets/icdar2015.py',
    '../../_base_/det_pipelines/wlnet_pipeline.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_ctw1500 = {{_base_.train_pipeline_ctw1500}}
test_pipeline_ctw1500 = {{_base_.test_pipeline_ctw1500}}

data = dict(samples_per_gpu=8,
            workers_per_gpu=8,
            val_dataloader=dict(samples_per_gpu=1),
            test_dataloader=dict(samples_per_gpu=1),
            train=dict(type='UniformConcatDataset', datasets=train_list, pipeline=train_pipeline_ctw1500),
            val=dict(type='UniformConcatDataset', datasets=test_list, pipeline=test_pipeline_ctw1500),
            test=dict(type='UniformConcatDataset', datasets=test_list, pipeline=test_pipeline_ctw1500))

evaluation = dict(interval=500, metric='hmean-iou')
