from html import entities
from data_utils import *

data = b_read_dataset('train_rel.jsonl')

data[0]

task = 'bid'

b_doccano_bak_train_dev(task)

train = b_read_dataset('train.json')


for entry in train:
    labels = entry['label']
    entities = []
    relations = []
    id = 1
    for start_offset,end_offset,label in labels:
        entity = {}
        entity['id'] = id
        id += 1
        entity['label'] = label
        entity['start_offset'] = start_offset
        entity['end_offset'] = end_offset
        entities.append(entity)
    entry['entities'] = entities
    # 如果entities中有 中标单位 和 中标金额
    # 如果entities中有 中标金额 和 中标金额单位
    # 则生成关系
    for entity in entities:
        id = 1
        if entity['label'] == '中标单位':
            for entity2 in entities:
                if entity2['label'] == '中标金额':
                    relation = {}
                    relation['id'] = id
                    id += 1
                    relation['from_id'] = entity['id']
                    relation['to_id'] = entity2['id']
                    relation['label'] = '中标单位-中标金额'
                    relations.append(relation)
        if entity['label'] == '中标金额':
            for entity2 in entities:
                if entity2['label'] == '中标金额单位':
                    relation = {}
                    relation['id'] = id
                    id += 1
                    relation['from_id'] = entity['id']
                    relation['to_id'] = entity2['id']
                    relation['label'] = '中标金额-中标金额单位'
                    relations.append(relation)
    entry['relations'] = relations

        

        
    