dataset_type = 'IcdarDataset'
data_root = '/home/atom/Research_STD/Datasets/mmocr/icdar2017'
train = dict(type=dataset_type, ann_file=f'{data_root}/instances_training.json', img_prefix=f'{data_root}', pipeline=None)

test = dict(type=dataset_type, ann_file=f'{data_root}/instances_val.json', img_prefix=f'{data_root}', pipeline=None)

train_list = [train]

test_list = [test]
