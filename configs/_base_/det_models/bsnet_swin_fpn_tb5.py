'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2021-12-21 20:18:27
LastEditors: Zhangzixu
LastEditTime: 2022-01-07 14:49:51
'''
cp_num = 5
bs_degree = 4
reconstr_points = 100

model = dict(type='BSNet',
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
                 add_extra_convs='on_output',  # use P5
                 num_outs=5,
                 relu_before_extra_convs=True),
             bbox_head=dict(type='BSHead_tb',
                            bs_degree=bs_degree,
                            cp_num=cp_num,
                            in_channels=256,
                            scales=(8, 16, 32),
                            loss=dict(type='BSLoss_tb',
                                      bs_degree=bs_degree, cp_num=cp_num),
                            postprocessor=dict(type='BSPostprocessor_tb',
                                               bs_degree=bs_degree,
                                               cp_num=cp_num,
                                               num_reconstr_points=reconstr_points,
                                               alpha=1.0,
                                               beta=2.0,
                                               score_thr=0.3)))
