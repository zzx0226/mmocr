'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2022-01-04 11:08:49
LastEditors: Zhangzixu
LastEditTime: 2022-01-04 14:05:34
'''
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[
                    58.395, 57.12, 57.375], to_rgb=True)

# for icdar2015
train_pipeline_icdar2015 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='LoadTextAnnotations', with_bbox=True,
         with_mask=True, poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 /
         255, saturation=0.5, contrast=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomScaling', size=800, scale=(3. / 4, 5. / 2)),
    # dict(type='RandomCropFlip', crop_ratio=0.5, iter_num=1, min_area_ratio=0.2),
    dict(type='RandomCropPolyInstances', instance_key='gt_masks',
         crop_ratio=0.8, min_side_ratio=0.3),
    dict(type='RandomRotatePolyInstances', rotate_ratio=0.5,
         max_angle=30, pad_with_fixed_color=False),
    dict(type='SquareResizePad', target_size=800, pad_ratio=0.6),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='Pad', size_divisor=32),
    dict(type='FCENetFcosTargets', fourier_degree=5),
    dict(type='CustomFormatBundle', visualize=dict(
        flag=False, boundary_key=None)),
    dict(type='FormatBundle_wl'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_fourier', 'gt_labels'])
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

train_pipeline_ctw1500 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='LoadTextAnnotations', with_bbox=True,
         with_mask=True, poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 /
         255, saturation=0.5, contrast=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomScaling', size=800, scale=(3. / 4, 5. / 2)),
    dict(type='RandomCropFlip', crop_ratio=0.5, iter_num=1, min_area_ratio=0.2),
    dict(type='RandomCropPolyInstances', instance_key='gt_masks',
         crop_ratio=0.8, min_side_ratio=0.3),
    dict(type='RandomRotatePolyInstances', rotate_ratio=0.5,
         max_angle=30, pad_with_fixed_color=False),
    dict(type='SquareResizePad', target_size=800, pad_ratio=0.6),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='Pad', size_divisor=32),
    dict(type='FCENetFcosTargets', fourier_degree=5),
    dict(type='CustomFormatBundle', visualize=dict(
        flag=False, boundary_key=None)),
    dict(type='FormatBundle_wl'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_fourier', 'gt_labels'])
]

# img_scale_ctw1500 = (1080, 736)

img_scale_ctw1500 = (800, 800)
test_pipeline_ctw1500 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='MultiScaleFlipAug',
         img_scale=img_scale_ctw1500,
         flip=False,
         transforms=[
             dict(type='Resize', img_scale=(800, 800), keep_ratio=False),
             #  dict(type='Resize', img_scale=(1280, 800), keep_ratio=True),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]
