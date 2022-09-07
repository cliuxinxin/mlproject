import pandas as pd
import copy
from data_utils import *

med_path = '../assets/med/'

df = pd.read_csv(med_path + 'train.csv', sep='\t')

labels = df['label'].unique()

label_dict = {}

for label in labels:
    label_dict[label] = 0

def calculate_cats(x):
    tmp = copy.deepcopy(label_dict)
    if x['label'] in label_dict:
        tmp[x['label']]  = 1
    return tmp

df['cats'] = df.apply(calculate_cats,axis=1)

# 随机排序
df = df.sample(frac=1).reset_index(drop=True)

df_train = df[:int(len(df)*0.7)]
df_dev = df[int(len(df)*0.7):]

b_save_df_datasets(df_train,'train.json')
b_save_df_datasets(df_dev,'dev.json')



