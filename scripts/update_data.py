from data_utils import b_doccano_bak_train_dev,b_read_dataset,b_save_df_datasets
import pandas as pd

tasks = ['tender','tendercats','bid','bidcats','contract','contractcats']

def is_cats(task):
    if task[-4:] == 'cats':
        return True
    else:
        return False

def get_tag(data):
    for entry in data:
        entry['cats'] = ','.join(entry['label'])
    return data

def cats_process(task):
    data = b_read_dataset(task + '_train_dev.json')
    data = get_tag(data)
    df = pd.DataFrame(data)
    df = df[['md5','cats']]
    b_save_df_datasets(df,task + '_train_dev.json')


for task in tasks:
    print("task:",task)
    b_doccano_bak_train_dev(task)
    if is_cats(task):
        cats_process(task)


