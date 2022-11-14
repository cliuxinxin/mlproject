from copy import deepcopy
from refac import *
import pandas as pd

client = get_doccano_client()

b_doccano_project_export(client,2,'train.json')

def cut_text(file='train.json',step_len=600,step_stride=300):
    data = b_read_dataset(file)

    new_data = []

    for sample in data:
        sample_len = len(sample['text'])

        for i in range(0,sample_len,step_stride):
            entry = {}
            entry['label'] = []

            entry['id'] = sample['id']

            start = i
            end = i + step_len

            entry['start'] = start
            entry['end'] = end

            if end > sample_len:
                end = sample_len
        
            entry['text'] = sample['text'][start:end]

            for label_start,label_end,label_ in sample['label']:
                if label_start >= start and label_end <= end:
                    entry['label'].append((label_start - start,label_end - start,label_))

            new_data.append(entry)

    b_save_dataset(new_data,file.replace('.json','.json'))

cut_text('train.json')
cut_text('dev.json')

data = b_read_dataset('train_cut.json')

new_data = []

def create_or_find(md5,data):
    for sample in data:
        if sample['md5'] == md5:
            return sample
    return {}

for sample in data:
    entry = create_or_find(sample['md5'],new_data)
    start = sample['start']
    end = sample['end']
    if entry == {}:
        # 复制sample 到 entry
        entry = deepcopy(sample)
        entry['label'] = []
        new_data.append(entry)
        for label_start,label_end,label_ in sample['label']:
            new_start = label_start + start
            new_end = label_end + start
            entry['label'].append((new_start,new_end,label_))
    else:
        # start end 之间的文字用新的覆盖
        entry['text'] = entry['text'][:start] + sample['text'] 
        for label_start,label_end,label_ in sample['label']:
            new_start = label_start + sample['start']
            new_end = label_end + sample['start']
            entry['label'].append((new_start,new_end,label_))
    # 去掉entry['label']中重复的标签
    entry['label'] = list(set(entry['label']))

b_save_dataset(new_data,'train_test.json')

len(new_data)

data = b_read_dataset('train_cut.json')

df = pd.DataFrame(data)

# 随机打乱数据
df = df.sample(frac=1).reset_index(drop=True)
length = len(df)
train_len = int(length * 0.8)
train_df = df[:train_len]
test_df = df[train_len:]
b_save_dataset(train_df,'train.json')
b_save_dataset(test_df,'dev.json')

data = b_read_dataset('train.json')

data = b_read_dataset('train_cut.json')

for entry in data:
    print(entry['start'])
