'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2022-03-12 09:59:46
LastEditors: Zhangzixu
LastEditTime: 2022-03-12 19:09:21
'''
_base_ = [
    '../../_base_/runtime_10e.py', '../../_base_/schedules/schedule_adam_finetune_200.py',
    '../../_base_/det_models/wlnet_r50dcnv2_fpn_test.py', '../../_base_/det_datasets/ctw1500TT.py',
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

evaluation = dict(interval=5, metric='hmean-iou')
