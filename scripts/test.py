from data_utils import *
from google_utils import *

task = 'bid'
# 下载标注好的文件,train_dev_label.json
gdrive_download_labeled_data()

data = b_read_dataset('train_dev_label.json')

data = b_read_dataset('train_dev.json')

data = b_read_dataset('train.json')


train = b_read_dataset('train.json')
dev = b_read_dataset('dev.json')

train = pd.DataFrame(train)
dev = pd.DataFrame(dev)

train['dataset'] = task + '_train'
dev['dataset'] = task + '_dev'

train_dev = pd.concat([train,dev])

b_updata_db_datasets(task,train_dev)
b_save_df_datasets(train_dev,'train_dev.json')