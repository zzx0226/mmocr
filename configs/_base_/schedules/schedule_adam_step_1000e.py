'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2021-12-27 20:32:37
LastEditors: Zhangzixu
LastEditTime: 2022-01-07 12:06:19
LastEditTime: 2022-01-07 12:06:19
'''
# optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[400, 900, 950])
total_epochs = 1000
