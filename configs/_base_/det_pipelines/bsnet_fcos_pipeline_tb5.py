'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2021-12-27 20:30:05
LastEditors: Zhangzixu
LastEditTime: 2022-01-05 18:11:05
'''
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
bs = 4
cp = 5
train_pipeline_icdar2015 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='LoadTextAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5, contrast=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomScaling', size=800, scale=(3. / 4, 5. / 2)),
    dict(type='RandomCropFlip', crop_ratio=0.5, iter_num=1, min_area_ratio=0.2),
    dict(type='RandomCropPolyInstances', instance_key='gt_masks', crop_ratio=0.8, min_side_ratio=0.3),
    dict(type='RandomRotatePolyInstances', rotate_ratio=0.5, max_angle=30, pad_with_fixed_color=False),
    dict(type='SquareResizePad', target_size=800, pad_ratio=0.6),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='Pad', size_divisor=32),
    dict(type='BSNetFcosTargets_tb', bs=bs, cp=cp),
    dict(type='CustomFormatBundle', visualize=dict(flag=False, boundary_key=None)),
    dict(type='FormatBundle_cp'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_cp', 'gt_labels'])
]

img_scale_icdar2015 = (2260, 2260)
test_pipeline_icdar2015 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='MultiScaleFlipAug',
         img_scale=img_scale_icdar2015,
         flip=False,
         transforms=[
             dict(type='Resize', img_scale=(1280, 800), keep_ratio=True),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]

# for ctw1500
train_pipeline_ctw1500 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='LoadTextAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5, contrast=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomScaling', size=800, scale=(3. / 4, 5. / 2)),
    dict(type='RandomCropFlip', crop_ratio=0.5, iter_num=1, min_area_ratio=0.2),
    dict(type='RandomCropPolyInstances', instance_key='gt_masks', crop_ratio=0.8, min_side_ratio=0.3),
    dict(type='RandomRotatePolyInstances', rotate_ratio=0.5, max_angle=30, pad_with_fixed_color=False),
    dict(type='SquareResizePad', target_size=800, pad_ratio=0.6),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='Pad', size_divisor=32),
    dict(type='BSNetFcosTargets_tb', bs=bs, cp=cp),
    dict(type='CustomFormatBundle', visualize=dict(flag=False, boundary_key=None)),
    dict(type='FormatBundle_cp'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_cp', 'gt_labels'])
]

img_scale_ctw1500 = (1080, 736)
test_pipeline_ctw1500 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='MultiScaleFlipAug',
         img_scale=img_scale_ctw1500,
         flip=False,
         transforms=[
             dict(type='Resize', img_scale=(1280, 800), keep_ratio=True),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]
