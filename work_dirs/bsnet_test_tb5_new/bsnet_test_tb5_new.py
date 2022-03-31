checkpoint_config = dict(interval=10)
log_config = dict(
    interval=5,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = 'work_dirs/bsnet_test_tb5_new/epoch_60.pth'
workflow = [('train', 1)]
optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[400, 900, 1300])
total_epochs = 1500
cp_num = 5
bs_degree = 4
reconstr_points = 100
model = dict(
    type='BSNet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=2, fallback_on_stride=False),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True,
        act_cfg=None),
    bbox_head=dict(
        type='BSHead_tb',
        bs_degree=4,
        cp_num=5,
        in_channels=256,
        scales=(8, 16, 32),
        loss=dict(type='BSLoss_tb_new', bs_degree=4, cp_num=5),
        postprocessor=dict(
            type='BSPostprocessor_tb_new',
            bs_degree=4,
            cp_num=5,
            num_reconstr_points=100,
            alpha=1.0,
            beta=2.0,
            score_thr=0.3)))
dataset_type = 'IcdarDataset'
data_root = '/home/atom/Research_STD/Datasets/mmocr/ctw1500'
train = dict(
    type='IcdarDataset',
    ann_file=
    '/home/atom/Research_STD/Datasets/mmocr/ctw1500/instances_training.json',
    img_prefix='/home/atom/Research_STD/Datasets/mmocr/ctw1500/imgs',
    pipeline=None)
test = dict(
    type='IcdarDataset',
    ann_file=
    '/home/atom/Research_STD/Datasets/mmocr/ctw1500/instances_test.json',
    img_prefix='/home/atom/Research_STD/Datasets/mmocr/ctw1500/imgs',
    pipeline=None)
train_list = [
    dict(
        type='IcdarDataset',
        ann_file=
        '/home/atom/Research_STD/Datasets/mmocr/ctw1500/instances_training.json',
        img_prefix='/home/atom/Research_STD/Datasets/mmocr/ctw1500/imgs',
        pipeline=None)
]
test_list = [
    dict(
        type='IcdarDataset',
        ann_file=
        '/home/atom/Research_STD/Datasets/mmocr/ctw1500/instances_test.json',
        img_prefix='/home/atom/Research_STD/Datasets/mmocr/ctw1500/imgs',
        pipeline=None)
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
leval_prop_range_icdar2015 = ((0, 0.4), (0.3, 0.7), (0.6, 1.0))
train_pipeline_icdar2015 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='ColorJitter',
        brightness=0.12549019607843137,
        saturation=0.5,
        contrast=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='RandomScaling', size=800, scale=(0.75, 2.5)),
    dict(
        type='RandomCropPolyInstances',
        instance_key='gt_masks',
        crop_ratio=0.8,
        min_side_ratio=0.3),
    dict(
        type='RandomRotatePolyInstances',
        rotate_ratio=0.5,
        max_angle=30,
        pad_with_fixed_color=False),
    dict(type='SquareResizePad', target_size=800, pad_ratio=0.6),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='Pad', size_divisor=32),
    dict(
        type='BSNetTargets_tb_new',
        bs_degree=4,
        cp_num=5,
        level_proportion_range=((0, 0.4), (0.3, 0.7), (0.6, 1.0))),
    dict(
        type='CustomFormatBundle',
        keys=['p3_maps', 'p4_maps', 'p5_maps'],
        visualize=dict(flag=False, boundary_key=None)),
    dict(type='Collect', keys=['img', 'p3_maps', 'p4_maps', 'p5_maps'])
]
img_scale_icdar2015 = (2260, 2260)
test_pipeline_icdar2015 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2260, 2260),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1280, 800), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
leval_prop_range_ctw1500 = ((0, 0.25), (0.2, 0.65), (0.55, 1.0))
train_pipeline_ctw1500 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='ColorJitter',
        brightness=0.12549019607843137,
        saturation=0.5,
        contrast=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='RandomScaling', size=800, scale=(0.75, 2.5)),
    dict(
        type='RandomCropFlip', crop_ratio=0.5, iter_num=1, min_area_ratio=0.2),
    dict(
        type='RandomCropPolyInstances',
        instance_key='gt_masks',
        crop_ratio=0.8,
        min_side_ratio=0.3),
    dict(
        type='RandomRotatePolyInstances',
        rotate_ratio=0.5,
        max_angle=30,
        pad_with_fixed_color=False),
    dict(type='SquareResizePad', target_size=800, pad_ratio=0.6),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='Pad', size_divisor=32),
    dict(
        type='BSNetTargets_tb_new',
        bs_degree=4,
        cp_num=5,
        level_proportion_range=((0, 0.25), (0.2, 0.65), (0.55, 1.0))),
    dict(
        type='CustomFormatBundle',
        keys=['p3_maps', 'p4_maps', 'p5_maps'],
        visualize=dict(flag=False, boundary_key=None)),
    dict(type='Collect', keys=['img', 'p3_maps', 'p4_maps', 'p5_maps'])
]
img_scale_ctw1500 = (800, 800)
test_pipeline_ctw1500 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(800, 800), keep_ratio=False),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='IcdarDataset',
                ann_file=
                '/home/atom/Research_STD/Datasets/mmocr/ctw1500/instances_training.json',
                img_prefix=
                '/home/atom/Research_STD/Datasets/mmocr/ctw1500/imgs',
                pipeline=None)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='LoadTextAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(
                type='ColorJitter',
                brightness=0.12549019607843137,
                saturation=0.5,
                contrast=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='RandomScaling', size=800, scale=(0.75, 2.5)),
            dict(
                type='RandomCropFlip',
                crop_ratio=0.5,
                iter_num=1,
                min_area_ratio=0.2),
            dict(
                type='RandomCropPolyInstances',
                instance_key='gt_masks',
                crop_ratio=0.8,
                min_side_ratio=0.3),
            dict(
                type='RandomRotatePolyInstances',
                rotate_ratio=0.5,
                max_angle=30,
                pad_with_fixed_color=False),
            dict(type='SquareResizePad', target_size=800, pad_ratio=0.6),
            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
            dict(type='Pad', size_divisor=32),
            dict(
                type='BSNetTargets_tb_new',
                bs_degree=4,
                cp_num=5,
                level_proportion_range=((0, 0.25), (0.2, 0.65), (0.55, 1.0))),
            dict(
                type='CustomFormatBundle',
                keys=['p3_maps', 'p4_maps', 'p5_maps'],
                visualize=dict(flag=False, boundary_key=None)),
            dict(
                type='Collect', keys=['img', 'p3_maps', 'p4_maps', 'p5_maps'])
        ]),
    val=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='IcdarDataset',
                ann_file=
                '/home/atom/Research_STD/Datasets/mmocr/ctw1500/instances_test.json',
                img_prefix=
                '/home/atom/Research_STD/Datasets/mmocr/ctw1500/imgs',
                pipeline=None)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 800),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(800, 800), keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='IcdarDataset',
                ann_file=
                '/home/atom/Research_STD/Datasets/mmocr/ctw1500/instances_test.json',
                img_prefix=
                '/home/atom/Research_STD/Datasets/mmocr/ctw1500/imgs',
                pipeline=None)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 800),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(800, 800), keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=500, metric='hmean-iou')
work_dir = './work_dirs/bsnet_test_tb5_new'
gpu_ids = range(0, 1)
