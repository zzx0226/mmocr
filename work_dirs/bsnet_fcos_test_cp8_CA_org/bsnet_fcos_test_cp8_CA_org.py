checkpoint_config = dict(interval=10)
log_config = dict(
    interval=5,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[200, 400])
total_epochs = 1500
cp_num = 8
bs_degree = 4
reconstr_points = 50
model = dict(
    type='BSNet_fcos',
    backbone=dict(
        type='mmdet.ResNet_CA',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=2, fallback_on_stride=False),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='BS_FCOSHead',
        bs_degree=4,
        cp_num=8,
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        postprocessor=dict(
            type='BSPostprocessor_fcos',
            bs_degree=4,
            cp_num=8,
            num_reconstr_points=50,
            alpha=1.0,
            beta=2.0,
            score_thr=0.1)),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
        classes=('text', )))
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
bs = 4
cp = 8
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
    dict(type='BSNetFcosTargets', bs=4, cp=8),
    dict(
        type='CustomFormatBundle',
        keys=['gt_bboxes', 'gt_cp'],
        visualize=dict(flag=False, boundary_key=None)),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_cp', 'gt_labels'])
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
    dict(type='BSNetFcosTargets', bs=4, cp=8),
    dict(
        type='CustomFormatBundle',
        visualize=dict(flag=False, boundary_key=None)),
    dict(type='FormatBundle_cp'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_cp', 'gt_labels'])
]
img_scale_ctw1500 = (1080, 736)
test_pipeline_ctw1500 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1080, 736),
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
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=6,
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
            dict(type='BSNetFcosTargets', bs=4, cp=8),
            dict(
                type='CustomFormatBundle',
                visualize=dict(flag=False, boundary_key=None)),
            dict(type='FormatBundle_cp'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_cp', 'gt_labels'])
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
                img_scale=(1080, 736),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(1280, 800), keep_ratio=True),
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
                img_scale=(1080, 736),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(1280, 800), keep_ratio=True),
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
evaluation = dict(interval=20, metric='hmean-iou')
work_dir = './work_dirs/bsnet_fcos_test_cp8_CA_org'
gpu_ids = range(0, 1)
