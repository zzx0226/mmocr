'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2021-12-28 21:37:13
LastEditors: Zhangzixu
LastEditTime: 2021-12-30 16:05:39
'''
cp_num = 8
bs_degree = 4
reconstr_points = 50

model = dict(
    type='BSNet_fcos',
    backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
        )),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=0,
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
        loss_cls=dict(type='FocalLoss', use_sigmoid=True,
                      gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(type='CrossEntropyLoss',
                             use_sigmoid=True, loss_weight=1.0),
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
