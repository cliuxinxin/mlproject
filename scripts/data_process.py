from data_utils import *
# from mysql_utils import *
# from google_utils import *

# gdrive_download_best_model()

# b_doccano_train_dev_update()

# t1 = time.time()
# b_label_dataset_multprocess('train_dev.json')
# t2 = time.time()
# t2 - t1

b_doccano_compare('train_dev.json','train_dev_label.json')
b_generate_cats_datasets_by_compare('train_dev.json','train_dev_label.json')