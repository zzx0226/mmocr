'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2022-02-15 16:34:48
LastEditors: Zhangzixu
LastEditTime: 2022-03-24 15:33:14
'''
wavelet_type = 'dmey'
reconstr_points = 50

model = dict(
    type='WLNet',
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
        out_indices=(1, 2, 3),
        with_cp=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
        )),
    neck=dict(type='mmdet.FPN',
              in_channels=[192, 384, 768],
              out_channels=256,
              add_extra_convs='on_output',
              num_outs=3,
              relu_before_extra_convs=True,
              act_cfg=None),
    bbox_head=dict(type='WLHead',
                   wavelet_type=wavelet_type,
                   in_channels=256,
                   scales=(8, 16, 32),
                   loss=dict(type='WLLoss_test', wavelet_type=wavelet_type),
                   postprocessor=dict(type='WLPostprocessor',
                                      num_reconstr_points=reconstr_points,
                                      wavelet_type=wavelet_type,
                                      alpha=1.0,
                                      beta=2.0,
                                      score_thr=0.3)))
