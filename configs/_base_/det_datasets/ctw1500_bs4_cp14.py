cp = 14
bs = 4

dataset_type = 'CurveDataset'
data_root = '/home/atom/Research_STD/Datasets/mmocr/ctw1500'

train = dict(type=dataset_type,
             ann_file=f'{data_root}/train_labels_with_bs_{bs}_cp_{cp}.json',
             img_prefix=f'{data_root}/imgs/training/',
             pipeline=None)

test = dict(type=dataset_type,
            ann_file=f'{data_root}/CTW_test_with_bs_{bs}_cp_{cp}.json',
            img_prefix=f'{data_root}/imgs/test/',
            pipeline=None)

train_list = [train]

test_list = [test]
