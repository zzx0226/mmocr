'''
Description: 
Version: 1.0
Autor: Zhangzixu
Date: 2022-01-02 18:46:09
LastEditors: Zhangzixu
LastEditTime: 2022-01-10 13:25:22
'''
# optimizer
optimizer = dict(type='SGD', lr=1e-4, momentum=0.90, weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=True)
total_epochs = 1500
