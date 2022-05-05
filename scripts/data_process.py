from data_utils import *
# from mysql_utils import *
from google_utils import *

gdrive_download_best_model()

b_doccano_train_dev_update()

t1 = time.time()
b_label_dataset_multprocess('train_dev.json')
t2 = time.time()
t2 - t1

b_doccano_compare('train_dev.json','train_dev_label.json')
b_generate_cats_datasets_by_compare('train_dev.json','train_dev_label.json')

b_remove_invalid_label('train_dev.json')

b_generate_compare_refine('train_dev.json','train_dev_remove.json')

gdrive_download_best_model_cats()

data = b_select_data_by_model('tender',300)

# text 改名 data
data.rename(columns={'text':'data'},inplace=True)

b_save_df_datasets(data,'train_dev_imp.json')

b_label_dataset_multprocess('train_dev_imp.json')

train_dev = b_read_dataset('train_dev_imp_label.json')

train_dev = pd.DataFrame(train_dev)

train,dev = b_split_train_test(train_dev,0.8)

b_save_df_datasets(train,'train.json')
b_save_df_datasets(dev,'dev.json')

# b_doccano_upload('train.json',2)
# b_doccano_upload('dev.json',3)






