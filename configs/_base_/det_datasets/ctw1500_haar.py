dataset_type = 'IcdarDataset'
wavelet_type = 'haar'

data_root = '/home/atom/Research_STD/Datasets/mmocr/ctw1500'

train = dict(type=dataset_type,
             ann_file=f'{data_root}/instances_training_with_{wavelet_type}.json',
             img_prefix=f'{data_root}/imgs',
             pipeline=None)

test = dict(type=dataset_type,
            ann_file=f'{data_root}/instances_test_with_{wavelet_type}.json',
            img_prefix=f'{data_root}/imgs',
            pipeline=None)

train_list = [train]

test_list = [test]
