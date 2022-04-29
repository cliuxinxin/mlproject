from data_utils import * 

train_dev = b_read_dataset('train_dev.json')

# 看看id是否重复
ids = []
for sample in train_dev:
    ids.append(sample['md5'])

# 查看id是否重复
len(set(ids))
len(ids)