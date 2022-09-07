from data_utils import *

task = 'bid'

b_doccano_bak_train_dev(task)

def select_data(data):
    new_data = []
    for entry in data:
        if entry['entities']:
            new_data.append(entry)
    return new_data


train = b_read_dataset('train.json')
dev = b_read_dataset('dev.json')
train = b_convert_ner_rel(train)
dev = b_convert_ner_rel(dev)
train = select_data(train)
dev = select_data(dev)
b_save_list_datasets(train,'train.json')
b_save_list_datasets(dev,'dev.json')





        
    