'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2021-12-27 20:32:37
LastEditors: Zhangzixu
LastEditTime: 2022-01-05 12:14:44
'''
# optimizer
optimizer = dict(type='Adam', lr=1.2e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=True)
total_epochs = 1500
