'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2021-12-21 20:18:27
LastEditors: Zhangzixu
LastEditTime: 2022-01-07 11:57:57
'''
cp_num = 5
bs_degree = 4
reconstr_points = 100

model = dict(type='BSNet',
             backbone=dict(type='mmdet.ResNet',
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
             neck=dict(type='mmdet.FPN',
                       in_channels=[512, 1024, 2048],
                       out_channels=256,
                       add_extra_convs='on_output',
                       num_outs=3,
                       relu_before_extra_convs=True,
                       act_cfg=None),
             bbox_head=dict(type='BSHead_tb',
                            bs_degree=bs_degree,
                            cp_num=cp_num,
                            in_channels=256,
                            scales=(8, 16, 32),
                            loss=dict(type='BSLoss_tb', bs_degree=bs_degree, cp_num=cp_num),
                            postprocessor=dict(type='BSPostprocessor_tb',
                                               bs_degree=bs_degree,
                                               cp_num=cp_num,
                                               num_reconstr_points=reconstr_points,
                                               alpha=1.0,
                                               beta=2.0,
                                               score_thr=0.3)))
