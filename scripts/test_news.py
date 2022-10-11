from data_utils import *

def get_unlabel_news(all, data,number=200):
    df_all = pd.DataFrame(all)
    df_data = pd.DataFrame(data)

    # 根据source去掉重复项
    df_all = df_all.drop_duplicates(subset=['source'], keep='first')
    # 随机
    df_all = df_all.sample(frac=1)

    df = df_all[~df_all.source.isin(df_data.source)][:number]

    b_save_df_datasets(df, 'news.json')

def get_ids(data):
    statc_ent = []
    for entry in data:
        for entity in entry['entities']:
            statc_ent.append(entity['id'])

    statc_rel = []
    for entry in data:
        for entity in entry['relations']:
            statc_rel.append(entity['id'])
    return statc_ent[-1] + 1, statc_rel[-1] + 1


def drop_empty(file):
    data = b_read_dataset(file)
    new_data = []
    for entry in data:
        if entry['entities']:
            new_data.append(entry)
    b_save_list_datasets(new_data,file)

def process_xlsx(file):
    df = pd.read_excel(ASSETS_PATH + file)
    df.columns = ['title', 'source', 'image', 'summery', 'labels', 'status', 'title_detail', 'pubdate', 'no_impot','writer', 'content']
    # 去掉为空的数据
    df = df.dropna(subset=['content'])
    # 去掉重复项
    df = df.drop_duplicates(subset=['source'], keep='first')
    df['text'] = df['title'] +'\n' + df['content']
    df.drop(columns=['content'], inplace=True)
    b_save_df_datasets(df,'news_all.json')

# 导出文件
def export_rel(num):
    b_doccano_export_project(1,'data.json')
    data = b_read_dataset('data.json')
    b_save_list_datasets(data[:num],'data.json')
    drop_empty('data.json')
    data = b_read_dataset('data.json')
    df = pd.DataFrame(data)
    # 随机
    df = df.sample(frac=1)
    # 保存训练集
    train_len = int(len(df) * 0.8)
    b_save_df_datasets(df[:train_len], 'train.json')
    b_save_df_datasets(df[train_len:], 'dev.json')

def process_entry(entry):
    ents  = {}
    text = entry['text']

    for ent in entry['entities']:
        ents[ent['id']] = text[ent['start_offset']:ent['end_offset']]

    events = []
    concepts = []
    views = []

    def find_ent_events(events,ent):
        for event in events:
            if event['ent'] == ent:
                return event
        return {}

    def process_records(records):
        for item in records:
            for key,value in item.items():
                item[key] = ents[value]
            item['url'] = entry['source']
            item['title'] = entry['title']
            item['pubdate'] = entry['pubdate']

    for relation in entry['relations']:
        relation_type = relation['type']
        if relation_type == '概念解释':
            concepts.append({
            'concept':relation['from_id'],
            'explation':relation['to_id']
        })
            continue
        if relation_type == '实体观点':
            views.append({
            'ent':relation['from_id'],
            'view':relation['to_id']
        })
            continue
        else:
            ent = relation['from_id']
            event = find_ent_events(events,ent)
            if event == {}: 
                events.append(event)
            event['ent'] = ent
            if relation_type == '实体事件':
                event['event'] = relation['to_id']
            if relation_type == '实体时间':
                event['time'] = relation['to_id']
            if relation_type == '实体地点':
                event['place'] = relation['to_id']

    process_records(events)
    process_records(concepts)
    process_records(views)

    return events,concepts,views

export_rel()

process_xlsx('test.xlsx')

# 所有数据 news_all.json
# 导出数据 train.json
all = b_read_dataset('news_all_label.json')
data = b_read_dataset('data.json')

get_unlabel_news(all, data,500)

ent_id,rel_id = get_ids(data)

task = 'news'
b_gpu_rel_label(task,'news.json',ent_id ,rel_id)

events = []
concepts = []
views = []

for entry in all:
    e,c,v = process_entry(entry)
    events.extend(e)
    concepts.extend(c)
    views.extend(v)

views = pd.DataFrame(views)
# 去掉重复
views = views.drop_duplicates(subset=['ent','view'], keep='first')
views.to_csv(ASSETS_PATH + 'views.csv',index=False)
concepts = pd.DataFrame(concepts)
# 去掉重复
concepts = concepts.drop_duplicates(subset=['concept','explation'], keep='first')
concepts.to_csv(ASSETS_PATH + 'concepts.csv',index=False)
events = pd.DataFrame(events)
# 去掉重复
events = events.drop_duplicates(subset=['ent','event'], keep='first')
events.to_csv(ASSETS_PATH + 'events.csv',index=False)

data = b_read_dataset('news_label.json')
df_data = pd.DataFrame(data)
df_data.drop(columns=['image','no_impot'], inplace=True)
b_save_df_datasets(df_data,'news_label_new.json')





            



