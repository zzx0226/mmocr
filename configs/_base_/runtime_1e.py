'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2022-03-12 09:03:48
LastEditors: Zhangzixu
LastEditTime: 2022-03-12 09:04:11
'''
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
