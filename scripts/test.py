from data_utils import *
from data_clean import *

task = 'bid'

b_doccano_train_dev_update(task)

train = b_read_dataset('train.json')
dev = b_read_dataset('dev.json')

col = project_configs[task]['col']
data_source = project_configs[task]['source']

cmp = b_read_dataset('compare.json')

for sample in cmp:
    sample['task'] = task
    # 如果sample[data_source] 不等于 NaN
    if len(sample[data_source]) <= 5 :
        sample['data_source'] = sample[data_source]
        sample['url'] = sample[data_source]
        del sample[data_source]

cmp = pd.DataFrame(cmp)
b_save_list_datasets(cmp,'compare_results.json')
