from data_utils import *

task = 'bid'


train_dev = b_read_dataset('train_dev.json')

for sample in train_dev:
    labels = sample['label']
    label_count = {}
    for label in labels:
        start = label[0]
        end = label[1]
        label_type = label[2]
        for label_orig in labels:
            orig_start = label_orig[0]
            orig_end = label_orig[1]
            orig_type = label_orig[2]
            if label_type != orig_type:
                if orig_end < start:
                    pass
                elif orig_start > end:
                    pass
                else:
                    print('-' * 20)
                    print(sample['dataset'])
                    print(sample['md5'])
                    print("{} {} {}".format(start, end, label_type))
                    print("{} {} {}".format(orig_start, orig_end, orig_type))



