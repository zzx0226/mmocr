'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2021-12-21 21:42:50
LastEditors: Zhangzixu
LastEditTime: 2022-01-10 20:33:35
'''
_base_ = [
    '../../_base_/runtime_10e.py',
    '../../_base_/schedules/schedule_adam_step_1500e_bsnet_tb.py',  # schedule_adam_step_1500e_bsnet_tb
    '../../_base_/det_models/bsnet_r50dcnv2_fpn_tb5_new.py',
    '../../_base_/det_datasets/totaltext.py',
    '../../_base_/det_pipelines/bsnet_pipeline_tb5_new.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_totaltext = {{_base_.train_pipeline_totaltext}}
test_pipeline_totaltext = {{_base_.test_pipeline_totaltext}}

data = dict(samples_per_gpu=8,
            workers_per_gpu=8,
            val_dataloader=dict(samples_per_gpu=1),
            test_dataloader=dict(samples_per_gpu=1),
            train=dict(type='UniformConcatDataset', datasets=train_list, pipeline=train_pipeline_totaltext),
            val=dict(type='UniformConcatDataset', datasets=test_list, pipeline=test_pipeline_totaltext),
            test=dict(type='UniformConcatDataset', datasets=test_list, pipeline=test_pipeline_totaltext))

evaluation = dict(interval=5, metric='hmean-iou')
#