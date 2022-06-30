from data_utils import * 
from recon.loaders import read_jsonl
from recon.types import Example
from recon.stats import get_ner_stats

task = 'bid'
b_doccano_train_dev_update(task)

data = b_read_dataset('train_dev.json')

data[0]

new_data = []

for entry in data:
    new_entry = {}
    text = entry['data']
    labels = entry['label']
    new_entry['text'] = text
    spans = []

    for start,end,label in labels:
        new_label = {}
        new_label['start'] = start
        new_label['end'] = end
        new_label['label'] = label
        spans.append(new_label)
    new_entry['spans'] = spans
    new_data.append(new_entry)

new_data[0]

b_save_list_datasets(new_data, 'train_debug.json')

data = read_jsonl(ASSETS_PATH + 'train_debug.json')

assert isinstance(data[0],Example)

get_ner_stats(data)