'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2021-12-27 20:32:37
LastEditors: Zhangzixu
LastEditTime: 2022-01-05 12:26:55
'''
# optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[500, 900, 1300])
total_epochs = 1500
