_base_ = [
    '../../_base_/runtime_10e.py', '../../_base_/schedules/schedule_adam_step_1500e_bsnet.py',
    '../../_base_/det_models/bsnet_r50dcnv2_fpn_bbox.py', '../../_base_/det_datasets/ctw1500.py',
    '../../_base_/det_pipelines/bsnet_pipeline_bbox.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_icdar2015 = {{_base_.train_pipeline_icdar2015}}
test_pipeline_icdar2015 = {{_base_.test_pipeline_icdar2015}}

data = dict(samples_per_gpu=8,
            workers_per_gpu=8,
            val_dataloader=dict(samples_per_gpu=1),
            test_dataloader=dict(samples_per_gpu=1),
            train=dict(type='UniformConcatDataset', datasets=train_list, pipeline=train_pipeline_icdar2015),
            val=dict(type='UniformConcatDataset', datasets=test_list, pipeline=test_pipeline_icdar2015),
            test=dict(type='UniformConcatDataset', datasets=test_list, pipeline=test_pipeline_icdar2015))

evaluation = dict(interval=5, metric='hmean-iou')
