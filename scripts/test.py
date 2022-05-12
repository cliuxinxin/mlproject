from data_utils import *
from data_clean import *
from datetime import datetime

task = 'bid'

b_doccano_train_dev_update(task)

train = b_read_dataset('train.json')
dev = b_read_dataset('dev.json')

col = project_configs[task]['col']
data_source = project_configs[task]['source']

cmp = pd.DataFrame(dev)
# 找到所有的na数据
na_data = cmp[cmp[col].isna()]

# 如果data_source 为 Nan，就用 source_website_address 填充
cmp['data_source'] = cmp['data_source'].fillna(cmp['source_website_address'])
cmp['task'] = cmp['task'].fillna('bid')
cmp['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
cmp.drop(['source_website_address'],axis=1,inplace=True)

# 将text改名为data
cmp.rename(columns={'text':'data'},inplace=True)

b_save_df_datasets(cmp,'train.json')
b_save_df_datasets(cmp,'dev.json')

b_doccano_upload_by_task('train.json',task,'train')
b_doccano_upload_by_task('dev.json',task,'dev')
