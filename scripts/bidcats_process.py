from data_utils import *

def add_md5(data):
    for entry in data:
        entry['text'] = entry['text'] + '@crazy@' + entry['md5']
    return data

def remove_md5(data):
    for entry in data:
        entry['text'] = entry['text'].split('@crazy@')[0]
    return data

def change_label_to_cats(data):
    for entry in data:
        entry['cats'] = entry['label']
    return data

def b_doccano_export_project_md5(project_id,filename):
    """
    导出项目数据，并且添加md5
    """
    b_doccano_export_project(project_id,filename)
    data = b_read_dataset(filename)
    data = remove_md5(data)
    return data

def b_doccano_export_cat_project_md5(project_id,file_name):
    # 导出分类项目，并且将分类项目的标签从cats的列表转换成字典
    data = b_doccano_export_project_md5(project_id,file_name)

    cats = b_doccano_export_cat_labels(project_id)

    stand_cats = {}
    for cat in cats:
        stand_cats[cat] = 0

    for entry in data:
        entry_stand_cats = copy.deepcopy(stand_cats)
        if 'cats' in entry:
            for cat in entry['cats']:
                entry_stand_cats[cat] = 1
            entry['cats'] = entry_stand_cats
        else:
            for cat in entry['label']:
                entry_stand_cats[cat] = 1
            entry['cats'] = entry_stand_cats

    b_save_list_datasets(data,file_name)

def b_save_list_datasets_md5(data,filename):
    """
    将dataframe保存为json文件
    """
    data = add_md5(data)
    b_save_list_datasets(data,filename)

def generate_doccano_url(x):
    return f'http://47.108.118.95:18000/projects/{x["project_id"]}/text-classification?page=1&q={x["md5"]}'

def generate_project_id(x):
    # 如果dataset
    train_dev = x['dataset'].split('_')[1]
    if train_dev == 'train':
        return 37
    elif train_dev == 'dev':
        return 38
    else:
        return 0


config = {
    'bidcats':{
        'train_id':37,
        'dev_id': 38,
},
    'tendercats':{
        'train_id':36,
        'dev_id': 36,
}
}

# task = 'tendercats'
task = 'bidcats'
threhold = 0.7

b_doccano_bak_train_dev(task)

data = b_read_dataset('bidcats_train_dev.json')

train_id = config[task]['train_id']
dev_id = config[task]['dev_id']

train = b_doccano_export_project_md5(train_id,'train.json')
dev = b_doccano_export_project_md5(dev_id,'dev.json')

b_doccano_export_cat_project_md5(train_id,'train.json')
b_doccano_export_cat_project_md5(dev_id,'dev.json')

df_train = pd.DataFrame(train)
df_dev = pd.DataFrame(dev)


df_train['project_id'] = train_id
df_dev['project_id'] = dev_id

data = b_read_dataset('train_tag.json')

df = pd.DataFrame(data)
df['project_id'] = train_id


df = pd.concat([df_train,df_dev])

data = b_read_dataset('bidcats_train_dev_tag.json')
df = pd.DataFrame(data)

df['project_id'] = df.apply(generate_project_id,axis=1)

df['doccano_url'] = df.apply(generate_doccano_url,axis=1)
# 把label 改成 cats ，吧source 改成 data_source
df['cats'] = df['label']
df['data_source'] = df['source']

df = df[['id','cats',  'tag', 'data_source','doccano_url']]
df.columns = ['id','human','ai','data_source','doccano_url']
df['is_same'] = df.apply(lambda x: x['human'] == x['ai'],axis=1)
df = df[['id',  'is_same','human', 'ai', 'data_source', 'doccano_url']]

b_save_df_datasets(df,'bidcats_tag.json')

df.to_csv(ASSETS_PATH + 'bidcats.csv',index=False)

def get_tag(doc,threhold):
    tag = []
    for label,prob in doc.cats.items():
        if prob > threhold:
            tag.append(label)
    if len(tag) == 0:
        tag = ['其他']
    return tag

nlp = b_load_best_model(task)

text = df_train.text.values.tolist()

docs = nlp.pipe(text)

tags = []
for doc in docs:
    tags.append(get_tag(doc,threhold))

df_train['tags'] = tags



tables = ['test_other_tender_bid_result','test_procurement_bid_result','test_tender_bid_result']

project_type = json.load(open(ASSETS_PATH + 'project_type.json'))

keywords = {}
for key,value in project_type.items():
    lable = key.split('_')[0]
    keyword = key.split('_')[1]
    if lable not in keywords:
        keywords[lable] = [keyword]
    else:
        keywords[lable].append(keyword)

keywords['EPC'] = ['EPC']

table = tables[1]

columns = ['title','detail_content']

key_words = []
for lable,keywords_list in keywords.items():
    for keyword in keywords_list:
        key_words.append(keyword)
dfs = []
for table in tables:
# 写出sql不包含key_words的数据
    sql = f"select id,source_website_address,detail_content from {table} where detail_content not regexp('{'|'.join(key_words)}') limit 300"

    df = mysql_select_df(sql)
    dfs.append(df)

df = pd.concat(dfs)
df = p_process_df(df,task)
# 根据md5值去重
df = df.drop_duplicates('md5')
text = df.text.values.tolist()

docs = nlp.pipe(text)
tags = []
for doc in docs:
    tags.append(get_tag(doc,threhold))
df['label'] = tags
df['table'] = table

train_len = int(len(df)*0.8)
b_save_df_datasets(df[:train_len],'train.json')
b_save_df_datasets(df[train_len:],'dev.json')