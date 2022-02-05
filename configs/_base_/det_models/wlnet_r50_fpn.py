wavelet_type = 'sym5'
reconstr_points = 50

model = dict(type='WLNet',
             backbone=dict(type='mmdet.ResNet',
                           depth=50,
                           num_stages=4,
                           out_indices=(1, 2, 3),
                           frozen_stages=-1,
                           norm_cfg=dict(type='BN', requires_grad=True),
                           init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
                           norm_eval=False,
                           style='pytorch'),
             neck=dict(type='mmdet.FPN',
                       in_channels=[512, 1024, 2048],
                       out_channels=256,
                       add_extra_convs='on_output',
                       num_outs=3,
                       relu_before_extra_convs=True,
                       act_cfg=None),
             bbox_head=dict(type='WLHead',
                            in_channels=256,
                            scales=(8, 16, 32),
                            loss=dict(type='WLLoss', wavelet_type=wavelet_type),
                            postprocessor=dict(type='WLPostprocessor',
                                               num_reconstr_points=reconstr_points,
                                               wavelet_type=wavelet_type,
                                               alpha=1.0,
                                               beta=2.0,
                                               score_thr=0.3)))
