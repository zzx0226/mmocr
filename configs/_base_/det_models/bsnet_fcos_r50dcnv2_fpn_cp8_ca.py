cp_num = 8
bs_degree = 4
reconstr_points = 50

model = dict(
    type='BSNet_fcos',
    backbone=dict(type='mmdet.ResNet_CA',
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
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='BS_FCOSHead',
        bs_degree=bs_degree,
        cp_num=cp_num,
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        #    loss=dict(type='BSLoss_fcos', bs_degree=4, cp_num=14),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        postprocessor=dict(type='BSPostprocessor_fcos',
                           bs_degree=bs_degree,
                           cp_num=cp_num,
                           num_reconstr_points=reconstr_points,
                           alpha=1.0,
                           beta=2.0,
                           score_thr=0.1)),
    train_cfg=None,
    test_cfg=dict(nms_pre=1000,
                  min_bbox_size=0,
                  score_thr=0.05,
                  nms=dict(type='nms', iou_threshold=0.5),
                  max_per_img=100,
                  classes=('text', )))
