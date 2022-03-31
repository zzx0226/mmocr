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
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-07, by_epoch=True)
total_epochs = 1500
cp_num = 8
bs_degree = 4
reconstr_points = 100
model = dict(
    type='BSNet_fcos',
    backbone=dict(
        type='mmdet.ResNet',
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
        start_level=0,
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
            num_reconstr_points=100,
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
cp = 14
bs = 4
dataset_type = 'CurveDataset'
data_root = '/home/atom/Research_STD/Datasets/mmocr/ctw1500'
train = dict(
    type='CurveDataset',
    ann_file=
    '/home/atom/Research_STD/Datasets/mmocr/ctw1500/train_labels_with_bs_4_cp_14.json',
    img_prefix='/home/atom/Research_STD/Datasets/mmocr/ctw1500/imgs/training/',
    pipeline=None)
test = dict(
    type='CurveDataset',
    ann_file=
    '/home/atom/Research_STD/Datasets/mmocr/ctw1500/CTW_test_with_bs_4_cp_14.json',
    img_prefix='/home/atom/Research_STD/Datasets/mmocr/ctw1500/imgs/test/',
    pipeline=None)
train_list = [
    dict(
        type='CurveDataset',
        ann_file=
        '/home/atom/Research_STD/Datasets/mmocr/ctw1500/train_labels_with_bs_4_cp_14.json',
        img_prefix=
        '/home/atom/Research_STD/Datasets/mmocr/ctw1500/imgs/training/',
        pipeline=None)
]
test_list = [
    dict(
        type='CurveDataset',
        ann_file=
        '/home/atom/Research_STD/Datasets/mmocr/ctw1500/CTW_test_with_bs_4_cp_14.json',
        img_prefix='/home/atom/Research_STD/Datasets/mmocr/ctw1500/imgs/test/',
        pipeline=None)
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
leval_prop_range_icdar2015 = ((0, 0.4), (0.3, 0.7), (0.6, 1.0))
train_pipeline_icdar2015 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='LoadTextAnnotations', with_bbox=True),
    dict(type='Load_bs_cp'),
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
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='Resize_cp'),
    dict(type='RandomFlip', flip_ratio=0.0, direction='horizontal'),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='FormatBundle_cp'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_cp'])
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
    dict(type='LoadTextAnnotations', with_bbox=True),
    dict(type='Load_bs_cp'),
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
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='Resize_cp'),
    dict(type='RandomFlip', flip_ratio=0.0, direction='horizontal'),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='FormatBundle_cp'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_cp'])
]
test_pipeline_ctw1500 = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
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
    samples_per_gpu=10,
    workers_per_gpu=6,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='CurveDataset',
                ann_file=
                '/home/atom/Research_STD/Datasets/mmocr/ctw1500/train_labels_with_bs_4_cp_14.json',
                img_prefix=
                '/home/atom/Research_STD/Datasets/mmocr/ctw1500/imgs/training/',
                pipeline=None)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(type='LoadTextAnnotations', with_bbox=True),
            dict(type='Load_bs_cp'),
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
            dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
            dict(type='Resize_cp'),
            dict(type='RandomFlip', flip_ratio=0.0, direction='horizontal'),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='FormatBundle_cp'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_cp'])
        ]),
    val=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='CurveDataset',
                ann_file=
                '/home/atom/Research_STD/Datasets/mmocr/ctw1500/CTW_test_with_bs_4_cp_14.json',
                img_prefix=
                '/home/atom/Research_STD/Datasets/mmocr/ctw1500/imgs/test/',
                pipeline=None)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
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
                type='CurveDataset',
                ann_file=
                '/home/atom/Research_STD/Datasets/mmocr/ctw1500/CTW_test_with_bs_4_cp_14.json',
                img_prefix=
                '/home/atom/Research_STD/Datasets/mmocr/ctw1500/imgs/test/',
                pipeline=None)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
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
work_dir = './work_dirs/bsnet_fcos_test'
gpu_ids = [1]
