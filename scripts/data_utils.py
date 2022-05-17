import configparser
import copy
import hashlib
import json
import math
import os
import pickle
import random
import re
import shutil
import time
import warnings
import zipfile
import queue
import threading
import glob
from copy import deepcopy
from datetime import datetime



import pandas as pd
import spacy
from doccano_api_client import DoccanoClient
from spacy.matcher import PhraseMatcher
from spacy.scorer import Scorer
from spacy.training import Example
from transformers import (AutoConfig, AutoModelForTokenClassification,
                          AutoTokenizer, pipeline)

from multiprocessing import Pool
from multiprocessing.managers import BaseManager


def d_parse_config():
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
    return config

project_configs = d_parse_config()



# 数据库导出字段 details
# doccano 导入字段 text label
# doccano 导出字段 data label
# label ： [[0,2,'O'],[1,3,'B-PER']]
# 分类标签 {"cats"：{"需要":1,"不需要":0}}

# 函数分为3个层次
# D Data manupulation 数据操作层
# P Preprocessing 数据预处理层
# B Building blocks 建构层

# 操作对象
# file - 直接的文件对象
# df - 数据表
# data - 列表

# ——————————————————————————————————————————————————
# 数据操作层
# ——————————————————————————————————————————————————

ROOT_PATH = '../'
ASSETS_PATH = ROOT_PATH + 'assets/'
DATA_PATH = ROOT_PATH + 'data/'
LOCK_FILE_PATH = ROOT_PATH + 'files_lock'
DATABASE_PATH = ROOT_PATH + 'database/'

# instantiate a client and log in to a Doccano instance
doccano_client = DoccanoClient(
    project_configs['doccano']['url'],
    project_configs['doccano']['user'],
    project_configs['doccano']['password']
)


# df保存jsonl文件
def d_save_df_datasets(df,path):
    with open(path,'w',encoding='utf-8') as f:
        for entry in df.to_dict('records'):
            json.dump(entry,f,ensure_ascii=False)
            f.write('\n')

# 数组保存jsonl文件
def d_save_list_datasets(data,path):
    with open(path,'w',encoding='utf-8') as f:
        for entry in data:
            json.dump(entry,f,ensure_ascii=False)
            f.write('\n')

# 读取数据集文件返回list
def d_read_json(path) -> list:
    data = []
    with open(path, 'r',encoding='utf8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 存储pkl
def d_save_pkl(value,path):
  with open(path, 'wb') as f:
    pickle.dump(value, f)

# 读取pkl
def d_read_pkl(path) -> object:
    with open(path, 'rb') as f:
        value = pickle.load(f)
    return value

def d_save_file(files,path):
    """
    写入平板文件，每行一个数据，以\n分隔
    """
    with open(path, 'w') as f:
        for file in files:
            f.write(file + '\n')

# 读取平板文件，返回每行内容，去掉内容中的\n
def d_read_file(path) -> list:
    with open(path, 'r') as f:
        data = f.readlines()
    data = [i.replace('\n', '') for i in data]
    return data



# ——————————————————————————————————————————————————
# 数据处理层
# ——————————————————————————————————————————————————

# 生成md5
def p_generate_md5(text) -> str:
    m = hashlib.md5()
    m.update(text.encode('utf-8'))
    return m.hexdigest()

# 预处理html文本
def p_html_text(df,column):
    df[column] = df[column].apply(p_filter_tags)


# 替换特殊字符
def p_replaceCharEntity(text) -> str:
    CHAR_ENTITIES = {'nbsp': ' ', '160': ' ',
                     'lt': '<', '60': '<',
                     'gt': '>', '62': '>',
                     'amp': '&', '38': '&',
                     'quot': '"', '34': '"', }

    re_charEntity = re.compile(r'&#?(?P<name>\w+);')
    sz = re_charEntity.search(text)
    while sz:
        entity = sz.group()  # entity全称，如&gt;
        key = sz.group('name')  # 去除&;后entity,如&gt;为gt
        try:
            text = re_charEntity.sub(CHAR_ENTITIES[key], text, 1)
            sz = re_charEntity.search(text)
        except KeyError:
            # 以空串代替
            text = re_charEntity.sub('', text, 1)
            sz = re_charEntity.search(text)
    return text

# 去掉网页标签
def p_filter_tags(text) -> str:
    # 先过滤CDATA
    re_cdata = re.compile('//<!\[CDATA\[[^>]*//\]\]>', re.I)  # 匹配CDATA
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)  # Script
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)  # style
    re_br = re.compile('<br\s*?/?>')  # 处理换行
    re_h = re.compile('</?\w+[^>]*>')  # HTML标签
    re_o = re.compile(r'<[^>]+>', re.S) # 其他
    re_comment = re.compile('<!--[^>]*-->')  # HTML注释
    s = re_cdata.sub('', text)  # 去掉CDATA
    s = re_script.sub('', s)  # 去掉SCRIPT
    s = re_style.sub('', s)  # 去掉style
    s = re_br.sub('\n', s)  # 将br转换为换行
    s = re_h.sub('', s)  # 去掉HTML 标签
    s = re_o.sub('',s)
    s = re_comment.sub('', s)  # 去掉HTML注释
    # 去掉多余的空行
    # blank_line = re.compile('\n+')
    # s = blank_line.sub('\n', s)
    s = p_replaceCharEntity(s)  # 替换实体
    return s

# 计算分割点
def p_cal_boder_end(block,length=500):
    borders_end = []
    # 计算出边界值
    for i in range(block):
        borders_end.append((i+1) * length)
    return borders_end

# 分割标签
def p_cal_border_and_label_belong(labels,borders_end):
    label_loc = []
    for label in labels:
        start = label[0]
        end = label[1]
        for idx,border in enumerate(borders_end):
            if start < border and end < border:
                label_loc.append(idx)
                break
            if start < border and end > border:
                pad = end - border + 20
                label_loc.append(idx)
                for idxc,border in enumerate(borders_end):
                    if idxc >= idx:
                        borders_end[idxc] += pad
                break
    return label_loc

# 拆分数据集
def p_generate_new_datasets(new_data, text, labels, borders_end, borders_start, label_loc,id):
    idx = 0
    for b_start,b_end in zip(borders_start,borders_end):
        entry = {}
        entry['data'] = text[b_start:b_end]
        new_labels = []
        for idxl,loc in enumerate(label_loc):
            if loc == idx:
                label = labels[idxl].copy()
                label[0] -= b_start
                label[1] -= b_start
                new_labels.append(label)
        entry['label'] = new_labels
        entry['id'] = id
        idx += 1
        if len(new_labels) != 0:
            new_data.append(entry)

def p_generate_cats_datasets(data:list):
    """
    传入生成cats标签的数据集,自动划成train_cats.json,dev_cats.json
    """
    pos = []
    neg = []
    for sample in data:
        if sample['cats']['需要'] == 1:
            pos.append(sample)
        else:
            neg.append(sample)


    # 随机排列pos，neg
    random.shuffle(pos)
    random.shuffle(neg)

    min_length = min(len(pos),len(neg))
    pos = pos[:min_length]
    neg = neg[:min_length]


    # train_cat,dev_cat
    train_cat = pos[:int(len(pos)*0.8)] + neg[:int(len(neg)*0.8)]
    dev_cat = pos[int(len(pos)*0.8):] + neg[int(len(neg)*0.8):]

    b_save_list_datasets(train_cat,'train_cats.json')
    b_save_list_datasets(dev_cat,'dev_cats.json')

def p_export_preprocess(path,task):
    """
    导出前处理，去掉md5信息，删除label读取信息
    """
    data = b_read_dataset(path)
    std_labels = b_read_db_labels(task)['label'].to_list()
    for entry in data:
        text = entry['data']
        text = text.split('@crazy@')[0]
        entry['data'] = text
        del entry['label_counts']
        for label in std_labels:
            del entry[label]
    b_save_list_datasets(data,path)


def p_upload_preprocess(file,task):
    # 预处理需要上传的数据
    data = b_read_dataset(file)
    std_labels = b_read_db_labels()
    std_labels = std_labels[std_labels['task'] == task]
    std_labels = std_labels['label'].tolist()
    for entry in data:
        text = entry['data']
        md5 = entry['md5']
        text  = text + '@crazy@' + md5
        entry['data'] = text
        try:
            labels = entry['label']
        except:
            labels = []
        entry['label_counts'] = len(labels)
        for label in std_labels:
            temp_list = []
            for sample_label in labels:
                if label == sample_label[2]:
                    label_text = text[sample_label[0]:sample_label[1]]
                    temp_list.append(label_text)
            entry[label] = ','.join(temp_list)
    b_save_list_datasets(data,file)

def p_process_df(df,task):
    """
    df 标准化文件处理
    """
    data_col = project_configs[task]['col']
    data_source_col = project_configs[task]['source']
    # 去掉空白行
    df = df[df[data_col].notnull()]
    df = df[df[data_source_col].notnull()]
    # 清洗html标签
    p_html_text(df,data_col)
    # 改名
    df.rename(columns={data_col:'data'},inplace=True)
    df.rename(columns={data_source_col:'source'},inplace=True)
    df['task'] = task
    df['md5'] = df[data_col].apply(p_generate_md5)
    df['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df.drop_duplicates(subset=['md5'],inplace=True)
    return df
# ——————————————————————————————————————————————————
# 构建层
# ——————————————————————————————————————————————————
def b_save_list_file(data:list,path):
    """
    将文件保存在assest目录下
    """
    path = ASSETS_PATH + path
    d_save_file(data,path)

def b_file_2_df(file_name,task) -> pd.DataFrame:
    """
    读取data里面的数据，清洗text文档，生成md5，根据configs.ini中的设置进行配置
    """
    df = pd.read_csv(DATA_PATH + file_name)
    df = p_process_df(df,task)
    return df

# 统计label个数
def b_label_counts(file) -> dict:
    # 统计train当中label的个数
    data = b_read_dataset(file)
    label_count = {}
    for entry in data:
        label = entry['label']
        for item in label:
            label_ = item[2]
            if label_ in label_count:
                label_count[label_] += 1
            else:
                label_count[label_] = 1
    return label_count


def b_save_db_datasets(df):
    """
    整体保存数据集的情况
    """
    d_save_pkl(df,DATABASE_PATH + 'datasets.pkl')


def b_read_db_datasets(task=''):
    """
    读取数据集的情况，如果传入task，可以取出该task的分布情况
    """
    datasets = d_read_pkl(DATABASE_PATH + 'datasets.pkl')
    if task:
        return datasets[datasets['dataset'].str.contains(task)]
    return d_read_pkl(DATABASE_PATH + 'datasets.pkl')


def b_save_db_basic(df):
    """
    整体保存基础数据
    """
    d_save_pkl(df,DATABASE_PATH + 'basic.pkl')

def b_read_db_basic(task=''):
    """
    整体读取基础数据，如果传入task，就取该task的数据
    """
    db = d_read_pkl(DATABASE_PATH + 'basic.pkl')
    if task:
        return db[db['task'] == task]
    return db


def b_save_db_labels(df):
  """
  保存label数据库
  """
  d_save_pkl(df,DATABASE_PATH + 'labels.pkl')

def b_read_db_labels(task=''):
  """
  读取label数据库
  """
  df = d_read_pkl(DATABASE_PATH + 'labels.pkl')
  if task:
    df = df[df['task'] == task]
  return df


def b_save_df_datasets(df,file):
    """
    df保存为数据集
    """
    d_save_df_datasets(df,ASSETS_PATH + file)

# 将data保存为datasets
def b_save_list_datasets(data,file):
    """
    list保存为数据集
    """
    d_save_list_datasets(data,ASSETS_PATH + file)


def b_load_best_model(task):
    """
    根据task读取模型
    """
    return spacy.load("../training/{}/model-best".format(task))


def b_load_best_cats():
    """
    读取cats模型
    """
    return spacy.load("../training/cats/model-best")

def b_read_text_file(file):
    """
    读取文本文件到list当中
    """
    data = d_read_file(ASSETS_PATH + file)
    return data

def b_read_dataset(file):
    """
    把dataset数据读取为list
    """
    data = d_read_json(ASSETS_PATH + file)
    return data

def b_updata_db_labels(task,df):
  """
  将某个task的labels的数据库更新
  """
  db = b_read_db_labels()
  db = db[~db['task'].str.contains(task)]
  db = pd.concat([db,df])
  b_save_db_labels(db)

def b_save_db_compare(df):
    """
    整体错题集的情况
    """
    d_save_pkl(df,DATABASE_PATH + 'compare.pkl')


def b_read_db_compare(task=''):
    """
    整体读取错题数据，如果传入task，就取该task的数据
    """
    db = d_read_pkl(DATABASE_PATH + 'compare.pkl')
    if task:
        return db[db['task'] == task]
    return db


# 分割数据集
def b_cut_datasets_size_pipe(file):
    data = b_read_dataset(file)

    new_data = []
    for entry in data:
        id = entry['id']
        text = entry['data']
        labels = entry['label']
        if len(text) < 500:
            new_data.append(entry)
        else:
            blcok = math.ceil(len(text) / 500)
            borders_end = p_cal_boder_end(blcok)
            borders_start = [0] + [i + 1 for i in borders_end[:-1]]
            label_loc = p_cal_border_and_label_belong(labels,borders_end)
            p_generate_new_datasets(new_data, text, labels, borders_end, borders_start, label_loc,id)
    return new_data

# 转换成百度excel格式，需要调整表头
def b_baidu_excel_format(file):
    new_data = b_cut_datasets_size_pipe(file)

    excel_data = []
    for entry in new_data:
        item = []
        item.append(entry['data'])
        labels = entry['label']
        for label in labels:
            loc = [label[0],label[1]]
            la = str(loc) +','+ label[2]
            item.append(la)
        excel_data.append(item)

    df = pd.DataFrame(excel_data)

    df.to_excel(file.split('.')[0] + '_new.xlsx',index=False)


# 从基础库中抽取未被业务未被抽样的数据
def b_extrct_data_from_db_basic(task) -> pd.DataFrame:
    db = b_read_db_basic(task)
    db_dataset = b_read_db_datasets(task)
    return db[db['md5'].isin(db_dataset['md5']) == False]


def b_select_data_by_model(task,num) -> list:
    """
    根据cats模型选择数据
    """
    db = b_extrct_data_from_db_basic(task)
    # 随机打乱顺序
    db = db.sample(frac=1).reset_index(drop=True)
    nlp = b_load_best_cats()

    sample_data = []
    for index,row in db.iterrows():
        text = row['data']
        if len(text) > 500:
            doc = nlp(text)
            print(doc.cats['需要'])
            if doc.cats['需要'] >= 0.9:
                sample_data.append(row)
            if len(sample_data) == num:
                break

    return pd.DataFrame(sample_data)

def b_doccano_upload(file,project_id):
    """
    把文件上传到doccano的项目中
    """
    doccano_client.post_doc_upload(project_id,file,ASSETS_PATH,column_data="data")


def b_doccano_upload_by_task(file,task,task_type):
    """
    根据文件，任务和任务类型上传到doccano
    """
    p_upload_preprocess(file,task)
    project_id = project_configs[task][task_type]
    b_doccano_upload(file,project_id)



def b_doccano_export_project(project_id,path,task=''):
    """
    doccano 导出数据
    """
    url = project_configs['doccano']['url']
    result = doccano_client.post(f'{url}/v1/projects/{project_id}/download', json={'exportApproved': False, 'format': 'JSONL'}) 
    task_id = result['task_id']
    while True:
        result = doccano_client.get(f'{url}/v1/tasks/status/{task_id}')
        if result['ready']:
            break
        time.sleep(1)
    result = doccano_client.get_file(f'{url}/v1/projects/{project_id}/download?taskId={task_id}')
    tmp_zip_path = ASSETS_PATH + '1.zip'
    with open(tmp_zip_path, 'wb') as f:
        for chunk in result.iter_content(chunk_size=8192): 
            f.write(chunk)
    zipfile.ZipFile(tmp_zip_path).extractall(path=ASSETS_PATH)
    shutil.move(ASSETS_PATH + 'all.jsonl', ASSETS_PATH + path)
    if task:
        p_export_preprocess(path,task)
    os.remove(tmp_zip_path)




# 删除项目中的数据
def b_doccano_delete_project(project_id):
    r = doccano_client.get_document_list(project_id)
    length = r['count']
    r = doccano_client.get_document_list(project_id,{'limit':[length],'offset':[0]})
    for entry in r['results']:
        doccano_client.delete_document(project_id,entry['id'])

# 在doccnano中查看某个标签的情况
# b_doccano_dataset_label_view('train.json',['招标项目编号'],1)
def b_doccano_dataset_label_view(file,labels,project_id):
    b_doccano_delete_project(project_id)
    train = b_read_dataset(file)
    new_train = []
    for entry in train:
        text = entry['data']
        for label in entry['label']:
            new_entry = copy.deepcopy(entry)
            new_entry.pop('data')

            start = label[0]
            end = label[1]
            s_start = start - 200 if start - 200 > 0 else 0
            s_end = end + 200 if end + 200 < len(text) else len(text)
            new_entry['data'] = text[s_start:s_end]
            label_ = [[start - s_start,end - s_start,label[2]]]
            new_entry['label'] = label_
            new_entry['s_start'] = s_start
            new_entry['s_end'] = s_end
            if label[2] in labels:
                new_train.append(new_entry)
    b_save_list_datasets(new_train,'train_new.json')
    b_doccano_upload('train_new.json',project_id)   

# b_cat_data_to_doccano(db,100,['招标编号','招标项目编号'])
# 找出一些关键词的数据
def b_doccano_cat_data(df,number,terms,project_id):

    number = number

    new_trian = []

    nlp = spacy.load('zh_core_web_lg')
    matcher = PhraseMatcher(nlp.vocab)
    terms = terms
    patterns = [nlp.make_doc(text) for text in terms]
    matcher.add("匹配条件", patterns)

    for index,row in df.iterrows():
        new_entry = row
        text = row['text']
        doc = nlp(text)
        matches = matcher(doc)
        labels = []
        for match_id, start, end in matches:
            span = doc[start:end]
            start = span.start_char
            end = span.end_char
            label =[start,end,'其他']
            labels.append(label)
        new_entry['label'] = labels
        new_trian.append(new_entry)
        if len(new_trian) == number:
            break
    df = pd.DataFrame(new_trian)
    df.rename(columns={'text':'data'},inplace=True)
    b_save_df_datasets(df,'train_cat.json')
    b_doccano_dataset_label_view('train_cat.json',['其他'],project_id)

def b_updata_db_datasets(task,df):
  """
  将某个task的datasets的数据库更新
  """
  db = b_read_db_datasets()
  db = db[~db['dataset'].str.contains(task)]
  db = pd.concat([db,df])
  b_save_db_datasets(db)


def b_doccano_init_dataseet(task,num,ratio):
    """
    根据任务，随机初始化数据集，并且上传到doccano
    """
    db = b_read_db_basic()
    # 随机抽取1000条数据
    db = db[db['task'] == task]
    db = db.sample(num)

    # 按照这2：8的比例切分训练和测试
    df_train,df_dev = b_split_train_test(db,ratio)

    # 分别保存到json文件中
    b_save_df_datasets(df_train,'train.json')
    b_save_df_datasets(df_dev,'dev.json')

    # 分别上传到doccano
    b_doccano_upload_by_task('train.json',task,'train')
    b_doccano_upload_by_task('dev.json',task,'dev')

    df_train['dataset'] = task + '_train'
    df_dev['dataset'] = task + '_dev'

    # 合并两个数据集
    df_train_dev = pd.concat([df_train,df_dev])

    b_updata_db_datasets(task,df_train_dev)


# 把demo的label更新到原来的file中
# b_change_label('train_dev.json','train_dev_label.json',['招标项目编号'])
def b_change_label(file,label_file,label_names):
    b_doccano_export_project(1,label_file)

    data = b_read_dataset(file)
    label_data = b_read_dataset(label_file)

    label_ids = [ entry['id'] for entry in label_data ]

    # 找到data中的id在label_ids中的数据，然后如果label[2]在label_names中的，就删除该label
    for entry in data:
        if entry['id'] in label_ids:
            label = entry['label']
            for l in label:
                if l[2] in label_names:
                    label.remove(l)
            entry['label'] = label

    for label_sample in label_data:
        id = label_sample['id']
        s_start = label_sample['start']
        for sample in data:
            if sample['id'] == id:
                break
        for label in label_sample['label']:
            if label[2] in label_names:
                start = label[0] + s_start
                end = label[1] + s_start
                sample['label'].append([start,end,label[2]])
    
    b_save_list_datasets(data,file)

# 转换json变成bio
def b_json2bio(file):
    '''
    将json文件中的数据转录为BIO形式，保存规则可以在43行修改
    '''
    file_name = file.split('.')[0]
    f_write = open(ASSETS_PATH + file_name + '_bio.txt', 'w', encoding='utf-8')
    load = b_read_dataset(file)
    for i in range(len(load)):
        labels = load[i]['label']
        text = load[i]['text']
        tags = ['O'] * len(text)
        for j in range(len(labels)):
            label = labels[j]
            tags[label[0]] = 'B-' + str(label[2])
            k = label[0]+1
            while k < label[1]:
                tags[k] = 'I-' + str(label[2])
                k += 1
        print(tags)
        for word, tag in zip(text, tags):
            f_write.write(word + '\t' + tag + '\n')
        f_write.write("\n")

# 将预测转换成json
def b_bio_to_json(text,predict):
    string="我是李明，我爱中国，我来自呼和浩特"
    predict=["o","o","i-per","i-per","o","o","o","b-loc","i-loc","o","o","o","o","b-per","i-loc","i-loc","i-loc"]
    item = {"string": string, "entities": []}
    entity_name = ""
    flag=[]
    visit=False
    for char, tag in zip(string, predict):
        if tag[0] == "b":
            if entity_name!="":
                x=dict((a,flag.count(a)) for a in flag)
                y=[k for k,v in x.items() if max(x.values())==v]
                item["entities"].append({"word": entity_name,"type": y[0]})
                flag.clear()
                entity_name=""
            entity_name += char
            flag.append(tag[2:])
        elif tag[0]=="i":
            entity_name += char
            flag.append(tag[2:])
        else:
            if entity_name!="":
                x=dict((a,flag.count(a)) for a in flag)
                y=[k for k,v in x.items() if max(x.values())==v]
                item["entities"].append({"word": entity_name,"type": y[0]})
                flag.clear()
            flag.clear()
            entity_name=""
    
    if entity_name!="":
        x=dict((a,flag.count(a)) for a in flag)
        y=[k for k,v in x.items() if max(x.values())==v]
        item["entities"].append({"word": entity_name,"type": y[0]})
    return item 

# 传入df，划分数据集
def b_split_train_test(df_db,ratio):
    df_train = df_db.sample(frac=ratio)
    df_test = df_db.drop(df_train.index)
    return df_train,df_test


def b_bio_labels_generate_from(task):
    """
    通过task生成bio标签 
    """
    labels = b_read_db_labels(task)
    labels = labels['label'].to_list()

    bio_labels = []
    for label in labels:
        bio_labels.append('B-'+label)
        bio_labels.append('I-'+label)
    bio_labels = ['O'] + bio_labels
    return bio_labels


# 将json标注改成数组对应标注法
# b_trans_dataset_bio(bio_labels,'train.json')
def b_bio_trans_dataset(bio_labels,file):
    file_name = file.split('.')[0]
    data  = b_read_dataset(file)
    
    new_data = []
    for sample in data:
        new_sample = {}
        text = sample['data']
        l_text = list(text)
        new_sample['data'] = l_text
        new_sample_label = [0] * len(l_text)
        for label in sample['label']:
            start = label[0]
            end = label[1]
            label_ = label[2]
            new_sample_label[start] = bio_labels.index('B-' + label_) 
            for i in range(start+1,end):
                new_sample_label[i] = bio_labels.index('I-' + label_)
        new_sample['label'] = new_sample_label
        new_data.append(new_sample)

    b_save_list_datasets(new_data,file_name + '_trf.json')

# 根据datasource查找原始数据
def b_find_orig_by_data_source(file,data_source):
    df = pd.read_csv(DATA_PATH + file)
    return df[df['data_source']==data_source].values[0]['details']

# doccano导出数据中不为空的数据整理出来
# b_get_all_label_data('train.jsonl')
def b_get_all_label_data(file):
    file_name = file.split('.')[0]

    train = b_read_dataset(file)

    new_train = []

    for sample in train:
    # 把label长度不为空的数据提取出来
        if len(sample['label']) != 0:
            new_train.append(sample)

    b_save_list_datasets(new_train,file_name + '.json')


def b_remove_invalid_label(file):
    """
    去掉标签中的空格字符，并且保存到_remove.json文件中
    """
    invalid_span_tokens = re.compile(r'\s')

    file_name = file.split('.')[0]

    data = b_read_dataset(file)

    cleaned_datas = []
    for sample in data:
        cleaned_data = deepcopy(sample)
        text = cleaned_data['data']
        labels = cleaned_data['label']
        clean_labels = []
        for start,end,label in labels:
            valid_start = start
            valid_end = end
        # if there's preceding spaces, move the start position to nearest character
            while valid_start < len(text) and invalid_span_tokens.match(
                text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                text[valid_end-1]):
                valid_end -= 1
            clean_labels.append([valid_start, valid_end, label])
        cleaned_data['label'] = clean_labels
        cleaned_datas.append(cleaned_data)  

    b_save_list_datasets(cleaned_datas,file_name + '_remove.json')

# 把bio数据集划分成最长的数据集,并且保存为train_trf_max.json
#split_dataset_by_max('train_trf.json',510) 
def b_bio_split_dataset_by_max(file,max_len):
    file_name = file.split('.')[0]
    max_length = max_len

    data = b_read_dataset(file)

    new_data = []
    for sample in data:
        data_text = sample["data"]
        data_label = sample["label"]

        divs = len(data_text)/max_length + 1

        every = len(data_text)// divs

        befor_after = (max_length - every ) // 2


        for i in range (0,int(divs)):
            new_sample = {}
            start  = i * every
            end = (i+1) * every
            if i == 0:
                end = end + befor_after * 2
            elif i == int(divs) - 1:
                start = start - befor_after * 2
            else:
                start = start - befor_after
                end = end + befor_after
            start = start if start >= 0 else 0
            end = end if end <= len(data_text) else len(data_text)
            start = int(start)
            end = int(end)
            new_text_data = data_text[start:end]
            new_label_data = data_label[start:end]
            new_sample["data"] = new_text_data
            new_sample["label"] = new_label_data
            new_data.append(new_sample)

    b_save_list_datasets(new_data,file_name  + '_maxlen.json')



# 合并train_dev数据，并且附加上meta，保存到train_dev.json
def b_combine_train_dev_meta():
    train = b_read_dataset('train.json')
    dev = b_read_dataset('dev.json')

    train_dev = train + dev

    df = pd.DataFrame(train_dev)

    db = b_read_db_datasets()

    df['md5'] = df['data'].apply(p_generate_md5)

    db_new = pd.merge(db,df,left_on='md5',right_on='md5',how='left')

    db_new = db_new.dropna()

    db_new.rename(columns={'data':'text'},inplace=True)

    b_save_df_datasets(db_new,'train_dev.json')



def b_split_train_dev(file):
    """
    将train_dev数据集划分为train,dev，并且保存为train.json,dev.json
    """
    train_dev = b_read_dataset(file)

    df_train_dev = pd.DataFrame(train_dev)

    train = df_train_dev[df_train_dev['dataset'] == 'tender_train']
    dev = df_train_dev[df_train_dev['dataset'] == 'tender_dev']

    b_save_df_datasets(train,'train.json')
    b_save_df_datasets(dev,'dev.json')

# 从doccano上面下载train，dev，合并保存到train_dev中
def p_doccano_download_tran_dev():
    b_doccano_update_train_dev()

    train = b_read_dataset('train.json')
    dev = b_read_dataset('dev.json')

    train = pd.DataFrame(train)
    dev = pd.DataFrame(dev)

    # 添加dataset列
    train['dataset'] = 'tender_train'
    dev['dataset'] = 'tender_dev'

    train_dev = pd.concat([train,dev])

    b_save_db_datasets(train_dev)

    b_save_df_datasets(train_dev,'train_dev.json')


def b_label_dataset_mult(task,file,thread_num):
    """
    指定文件和线程数，对文件进行标注，标注完成以后，保存成原文件。
    """
    def work(q,nlp,new_data):
        while True:
            if q.empty():
                return
            else:
                sample = q.get()
                try:
                    text = sample['data']
                except:
                    text = sample['text']
                doc = nlp(text)
                label = [[ent.start_char,ent.end_char,ent.label_] for ent in doc.ents]
                sample['label'] = label
                new_data.append(sample)

    file_name = file.split('.')[0]
    data = b_read_dataset(file)

    nlp = b_load_best_model(task)

    new_data = []
    q = queue.Queue()
    for sample in data:
        q.put(sample)
    thread_num = thread_num
    threads = []
    for i in range(thread_num):
        t = threading.Thread(target=work, args=(q,nlp,new_data))
        # args需要输出的是一个元组，如果只有一个参数，后面加，表示元组，否则会报错
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
        
    b_save_list_datasets(data,file_name+'_label.json')

# 使用机器标注好的数据进行对比
def b_compare_human_machine_label():
    train_dev_label = b_read_dataset('train_dev_mlabel.json')

    b_doccano_delete_project(1)

    new_data = []

    def record_data(new_data,wrong_type,label_type,sample, label):
        new_sample = copy.deepcopy(sample)
        new_sample.pop('data')
        new_sample.pop('mlabel')

        text = sample['data']

        new_sample['错误种类'] = wrong_type
        new_sample['标注人'] = label_type

        start = label[0]
        end = label[1]
        s_start = start - 200 if start - 200 > 0 else 0
        s_end = end + 200 if end + 200 < len(text) else len(text)

        new_sample['data'] = text[s_start:s_end] + '\n' + '错误类型'+label[2]
        label_ = [[start - s_start,end - s_start,label[2]]]
        new_sample['label'] = label_
        new_sample['s_start'] = s_start
        new_sample['s_end'] = s_end

        new_data.append(new_sample)

    for sample in train_dev_label:
        labels = sample['label']
        mlabels = sample['mlabel']

        wrong_mlabels = []
        for label in labels:
            if label not in mlabels:
                start = label[0]
                end = label[1]
                label_type = label[2]
                for mlabel in mlabels:
                    if mlabel[0] >= start:
                        break
                mstart = mlabel[0]
                mend = mlabel[1]
                if start == mstart and end == mend:
                    record_data(new_data,'机器错标类别','机器',sample,mlabel)
                    record_data(new_data,'机器错标类别答案','人',sample,label)
                
                    wrong_mlabels.append(mlabel)
                elif abs(start - mstart) < 10:
                    record_data(new_data,'机器错标位置','机器',sample,mlabel)
                    record_data(new_data,'机器错标位置答案','人',sample,label)
                    wrong_mlabels.append(mlabel)
                else:
                    record_data(new_data,'机器漏标答案','人',sample,label) 
                
        for mlabel in mlabels:
            if mlabel not in labels:
                if mlabel not in wrong_mlabels:
                    record_data(new_data,'机器多标','机器',sample,mlabel)  
    
    b_save_list_datasets(new_data,'train_dev_mlabel_new.json')
    b_doccano_upload('train_dev_mlabel_new.json',1)


def b_bio_datasets_trans_and_max():
    """
    将现有数据集转换成bio格式,并且切断成512长度
    
    输入文件：
    train.json
    dev.json

    生成文件：
    labels.txt
    train_trf.json bio格式数据
    dev_trf.json bio格式数据
    train_trf_maxlen 切断511长数据
    dev_trf_maxlen 切断511长数据
    """
    bio_labels = b_bio_labels_generate_from('labels.txt')

    b_bio_trans_dataset(bio_labels,'train.json')
    b_bio_trans_dataset(bio_labels,'dev.json')

    b_bio_split_dataset_by_max('train_trf.json',511)
    b_bio_split_dataset_by_max('dev_trf.json',511)


def b_trf_load_model():
    """
    读取trf模型，需要传入模型的地址

    前提条件需要在assets文件夹中放入labels.txt
    
    """
    path = 'bert-base-chinese-finetuned-ner/checkpoint'

    bio_labels = b_bio_labels_generate_from('labels.txt')

    config = AutoConfig.from_pretrained(path,
                                    num_labels=len(bio_labels),
                                    id2label={i: label for i, label in enumerate(bio_labels)},
                                    label2id={label: i for i, label in enumerate(bio_labels)}
)

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForTokenClassification.from_pretrained(path,config=config)
 
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    return nlp

def b_trf_label_dataset(nlp,file):
    """
    使用nlp模型对file文件进行标注,标签放在label字段下
    
    并且生成原文件名+_trf_label.json文件
    """
    file_name = file.split('.')[0]

    data = b_read_dataset(file)

    for sample in data:
        text = sample['data']
    
    # 把text按照500字符分割
        if len(text) > 500:
            text_list = [text[i:i+500] for i in range(0, len(text), 500)]
        else:
            text_list = [text]

        l_ents = nlp(text_list)

        for i,ents in enumerate(l_ents):
            s_start = i * 500
            for ent in ents:
                ent['start'] += s_start
                ent['end'] += s_start

    # 将ents中的list去掉
        result = []

        for ent in l_ents:
            result += ent

        new_labels = []
        new_label = {}
        for ent in result:
            if ent['entity'][0] == 'B':
                if new_label :
                # new_label['text'] = text[new_label['start']:new_label['end']]
                    new_label = [new_label['start'],new_label['end'],new_label['label']]
                    new_labels.append(new_label)
                    new_label = {}
                new_label['label'] = ent['entity'][2:]
                new_label['start']= ent['start']
                new_label['end'] = ent['end']
            elif ent['entity'][0] == 'I':
                if not new_label:
                    new_label['label'] = ent['entity'][2:]
                    new_label['start']= ent['start']
                    new_label['end'] = ent['end']
                if new_label['label'] == ent['entity'][2:]:
                    new_label['end'] = ent['end']
        sample['label'] = new_labels

    b_save_list_datasets(data, file_name +  '_trf_label.json')

def b_eavl_dataset(org_dataset_file,prd_dataset_file):
    """
    评估两个数据集的指标

    org_dataset_file : 原始数据集,label是标注
    prd_dataset_file : 预测数据集,label是预测
    
    """
    org_data = b_read_dataset(org_dataset_file)
    prd_data = b_read_dataset(prd_dataset_file)
    nlp = spacy.blank('zh')


    def get_ents(text, doc,sample,label_name='label'):
        ents = []   
        for start, end, label in sample[label_name]:
            span = doc.char_span(start, end, label=label,alignment_mode="contract")
            if span is None:
                msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
                warnings.warn(msg)
            else:
                ents.append(span)
        return ents


    examples = []
    for org,prd in zip(org_data,prd_data):
        text = org['data']
        org_doc = nlp.make_doc(text)
        prd_doc = nlp.make_doc(text)
        org_ents = get_ents(text, org_doc,org)
        prd_ents = get_ents(text, prd_doc,prd)
        org_doc.ents = org_ents
        prd_doc.ents = prd_ents
        example = Example(prd_doc,org_doc)
        examples.append(example)

    scorer = Scorer()
    scores = scorer.score(examples)

    ents_scores = scores['ents_per_type']

    df = pd.DataFrame(ents_scores)

    # 行列倒置
    df = df.T
    return df



def b_doccano_bak_train_dev(task):
    """
    根据task将的数据到task_train_bak.json和task_dev_bak.json中
    
    """
    b_doccano_export_project(project_configs[task]['train'],task + '_train_bak.json',task)
    b_doccano_export_project(project_configs[task]['dev'],task + '_dev_bak.json',task)


def b_process_origin_data():
    """
    讲mysql中导出的数据进行处理，生成basic库
    """
    db = b_file_2_df('Untitled.csv')

    b_save_db_basic(db)

def b_doccano_update_train_dev(task):
    """
    根据task将从doccano中下载到train.json和dev.json中
    
    """
    b_doccano_export_project(project_configs[task]['train'],'train.json',task)
    b_doccano_export_project(project_configs[task]['dev'],'dev.json',task)


def b_doccano_train_dev_update(task):
    """
    根据task从doccano的数据下载到train.json,项目3的数据到dev.json，并且合并到train_dev.json
    """
    b_doccano_update_train_dev(task)

    train = b_read_dataset('train.json')
    dev = b_read_dataset('dev.json')

    train = pd.DataFrame(train)
    dev = pd.DataFrame(dev)

    train['dataset'] = task + '_train'
    dev['dataset'] = task + '_dev'

    train_dev = pd.concat([train,dev])

    b_updata_db_datasets(task,train_dev)
    b_save_df_datasets(train_dev,'train_dev.json')

def b_generate_cats_by_label(target_labels=['报名开始时间','报名结束时间','预算','开标时间']):
    """
    根据是否有标签来提取数据，范围是train_dev,条件是标签在target_labels中
    """
    data =  b_read_dataset('train_dev.json')

    for sample in data:
        labels = sample['label']
        sample['cats'] = {"需要":0,"不需要":1} 
        for label in labels:
            label_ = label[2]
            if label_ in target_labels:
                sample['cats'] = {"需要":1,"不需要":0}
                break

    p_generate_cats_datasets(data)

def b_generate_cats_datasets_by_compare(org_file,cmp_file):
    """
    之前需要运行 b_label_dataset_multprocess 生成 train_dev_label.json
    根据对比情况，生成cats数据集
    
    """
    org_data = b_read_dataset(org_file) 
    cmp_data = b_read_dataset(cmp_file)

    for o_sample,c_sample in zip(org_data,cmp_data):
        text = o_sample['data']
        labels = o_sample['label']
        predicts = c_sample['label']
        o_sample['cats'] = {"需要":0,"不需要":1} 
        for entry in labels:
            if entry not in predicts:
                o_sample['cats'] = {"需要":1,"不需要":0}
                break
        for entry in predicts:
            if entry not in labels:
                o_sample['cats'] = {"需要":1,"不需要":0}
                break
    p_generate_cats_datasets(org_data)


def b_doccano_compare(org_file,cmp_file):
    """
    org_file 是人表的数据，cmp_file是机器标的数据。
    """
    org = b_read_dataset(org_file)
    cmp = b_read_dataset(cmp_file)

    new_data = []

    def record_data(new_data,wrong_type,label_type,sample, label):
        new_sample = copy.deepcopy(sample)
        new_sample.pop('data')

        text = sample['data']

        new_sample['wrong_type'] = wrong_type
        new_sample['labeler'] = label_type
        new_sample['label_type'] = label[2]

        start = label[0]
        end = label[1]
        s_start = start - 200 if start - 200 > 0 else 0
        s_end = end + 200 if end + 200 < len(text) else len(text)

        new_sample['data'] = text[s_start:s_end] + '\n' + '错误类型'+label[2] + '错误种类' + wrong_type
        label_ = [[start - s_start,end - s_start,label[2]]]
        new_sample['label'] = label_
        new_sample['s_start'] = s_start
        new_sample['s_end'] = s_end

        new_data.append(new_sample)

    for o_sample,c_sample in zip(org,cmp):
        labels = o_sample['label']
        mlabels = c_sample['label']

        wrong_mlabels = []
    
        label_dict = {}
        mlabel_dict = {}

        for label in labels:
            label_type = label[2]
            if label_type not in label_dict:
                label_dict[label_type] = 1
            else:
                label_dict[label_type] += 1
                record_data(new_data,'人标签重复','人',o_sample,label)
            if label not in mlabels:
                find_label = False
                for mlabel in mlabels:
                    mlabel_type = mlabel[2]
                    if label_type == mlabel_type:
                       record_data(new_data,'机器错标位置','机器',c_sample,mlabel)
                       record_data(new_data,'机器错标位置答案','人',o_sample,label)
                       find_label = True
                       wrong_mlabels.append(mlabel)
                       break
                if not find_label:
                    record_data(new_data,'机器漏标答案','人',o_sample,label)
            
        for mlabel in mlabels:
            mlabel_type = mlabel[2]
            if mlabel_type not in mlabel_dict:
                mlabel_dict[mlabel_type] = 1
            else:
                mlabel_dict[mlabel_type] += 1
                record_data(new_data,'机器标签重复','机器',c_sample,label)
            if mlabel not in labels:
                if mlabel not in wrong_mlabels:
                    record_data(new_data,'机器多标','机器',c_sample,mlabel) 
                    record_data(new_data,'机器多标修正','人',c_sample,mlabel) 

    b_save_list_datasets(new_data,'compare_imp.json')
    b_doccano_delete_project(1)
    b_doccano_upload('compare_imp.json',1)

def b_combine_compare_to_train_dev():
    """
    读取train和compare文件，根据compare中人类的答案进行修改
    """
    train_dev = b_read_dataset('train_dev.json')
    compare = b_read_dataset('compare.json')

    def find_sample_by_id(id):
        for sample in train_dev:
            if sample['id'] == id:
                return sample

    def find_label_by_type(labels,label_type):
        for idx,label in labels:
            if label[2] == label_type:
                return idx

    def is_delete_label(label,org_label_idx,org_sample):
        if label == []:
            org_sample['label'].pop(org_label_idx)

    def is_replace_label(label,org_label,org_label_idx,org_sample):
        if label != org_label:
            org_sample['label'][org_label_idx] = label


    for sample in compare:
        wrong_type = sample['wrong_type']
        id = sample['id']
        s_start = sample['start']
        label = sample['label']
        label_type = sample['labeler']
        org_sample = find_sample_by_id(id)
        org_label_idx = find_label_by_type(org_sample['labels'],label_type)
        org_label = org_sample['labels'][org_label_idx]
        if wrong_type in ['人标签重复','机器错标位置答案','机器漏标答案','机器多标修正']:
            is_delete_label(label,org_label_idx,org_sample)
            is_replace_label(label,org_label,org_label_idx,org_sample)


    b_save_list_datasets(train_dev,'train_dev.json')

def b_label_dataset_multprocess(task,file):
    """
    读取文件，并且标注好，把标注好的文件存为新的_label.json文件    
    """
    file_name = file.split('.')[0]

    data = b_read_dataset(file)

    class PoolCorpus(object):

        def __init__(self):
            self.nlp = b_load_best_model(task)
            self.data = []

        def add(self, sample):
            doc = self.nlp(sample['data'])
            labels = [[ent.start_char,ent.end_char,ent.label_] for ent in doc.ents]
            sample['label'] = labels
            self.data.append(sample)

        def get(self):
            return self.data

    BaseManager.register('PoolCorpus', PoolCorpus)

    with BaseManager() as manager:
        corpus = manager.PoolCorpus()

        with Pool() as pool:
            pool.map(corpus.add, (sample for sample in data))

        new_data = corpus.get()
    result = []
    for sample in data:
        md5 = sample['md5']
        new_sample = next(sample for sample in new_data if sample['md5'] == md5)
        result.append(new_sample)

    b_save_list_datasets(result,file_name+'_label.json')

def b_anlysis_rule(text,label,predict):
    """
    规则分析结果
    """
    label_start,label_end = label[:2]
    predict_start,predict_end = predict[:2]
    label_text = text[label[0]:label[1]]
    pred_text = text[predict[0]:predict[1]]
    result = "未分析成功"
    if label_text.strip() == pred_text.strip():
        if label_start == predict_start and label_end - predict_end >= 1:
            return '标签和预测结果相同，但是标签后面有空格'
        if label_start == predict_start and label_end - predict_end <= -1:
            return '标签和预测结果相同，但是预测后面有空格'
        if label_start - predict_start >= 1 and label_end == predict_end:
            return '标签和预测结果相同，但是预测前面有空格'
        if label_start - predict_start <= -1 and label_end == predict_end:
            return '标签和预测结果相同，但是标签前面有空格'
        else:
            return '标签和预测结果相同，但是标注位置不同'
    if label_start == predict_start and label_end - predict_end >= 1:
        return '预测后面少标' + str(label_end - predict_end) + '个字符'
    if label_start == predict_start and label_end - predict_end <= -1:
        return '预测后面多标' + str(predict_end - label_start) + '个字符'
    if label_start - predict_start >= 1 and label_end == predict_end:
        return '预测前面少标' + str(label_start - predict_start) + '个字符'
    if label_start - predict_start <= -1 and label_end == predict_end:
        return '预测前面多标' + str(predict_start - label_end) + '个字符'
    return result

def b_generate_compare_refine(task,org_file,cmp_file):
    """
    通过train_dev.json,train_dev_label.json 生成compare_results 用于 google refine处理使用
    """
    org_data = b_read_dataset(org_file)
    cmp_data = b_read_dataset(cmp_file)

    train_project_id = project_configs[task]['train']
    dev_project_id = project_configs[task]['dev']

    def record_data(result,label_type,text,label,predict,wrong_type=''):
        if wrong_type == 'AI多标':
            label = [0,0,label_type]
        if wrong_type == 'AI漏标':
            predict = [0,0,label_type]
        result['human_start'] = label[0]
        result['human_end'] = label[1]
        result['ai_start'] = predict[0]
        result['ai_end'] = predict[1]
        result['label_type'] = label_type
        result['human_label'] = text[label[0]:label[1]]
        result['ai_label'] = text[predict[0]:predict[1]]
        result['rule_anlysis'] = b_anlysis_rule(text,label,predict)
        if wrong_type == 'AI多标' or wrong_type == 'AI漏标':
            result['rule_anlysis'] = wrong_type
        dataset = train_project_id if result['dataset'] == task + '_train' else dev_project_id
        md5 = result['md5']
        
        if wrong_type:
            result['wrong_type'] = wrong_type
        if wrong_type == 'AI错标' and abs(label[0] - predict[0]) < 5:
            result['wrong_type'] = 'AI错标位置'
        if wrong_type == 'AI多标':
            result['human_start'] = result['human_end'] = result['human_label'] = ''
        if wrong_type == 'AI漏标':
            result['ai_start'] = result['ai_end'] = result['ai_label'] = ''

        result['is_human_correct'] = ''
        result['doccano_url'] = 'http://47.108.218.88:18000/projects/{}/sequence-labeling?page=1&q={}'.format(dataset,md5) 
        result['url'] = result['data_source'] + "#:~:text=" + (result['ai_label'] if result['ai_label'] else result['human_label'])
        new_result = deepcopy(result)
        del new_result['data_source']
        del new_result['id']
    
        results.append(new_result)



    results = []
    for sample in org_data:
        result = deepcopy(sample)
        del result['data']
        del result['label']
        del result['time']
        id = sample['id']
        md5 = sample['md5']
        text = sample['data']
        for entry in cmp_data:
            if entry['md5'] == md5:
                break
    
        labels = sample['label']
        predicts = entry['label']

        label_types = [ label[2] for label in labels ]
        predict_types = [ label[2] for label in predicts ]

        label_count = {}

        for label in labels:
            label_type = label[2]
            if label_type not in label_count:
                label_count[label_type] = 0
            label_count[label_type] += 1
    
        for label_type in label_count:
            if label_count[label_type] > 1:
                for label in labels:
                    if label[2] == label_type:
                        break
                temp_labels = deepcopy(labels)
                temp_labels.remove(label)
                for predict in temp_labels:
                    if predict[2] == label_type:
                        break
                record_data(result,label_type,text,label,predict,'手工重复标注')
    
        for label in labels:
            label_type = label[2]
            if label_type not in predict_types:
                record_data(result,label_type,text,label,label,'AI漏标')

        for predict in predicts:
            predict_type = predict[2]
            if predict_type not in label_types:
                record_data(result,predict_type,text,predict,predict,'AI多标')

        for label in labels:
            if label not in predicts:
                label_type = label[2]
                for predect in predicts:
                    if predect[2] == label_type:
                        record_data(result,label_type,text,label,predect,'AI错标')
    
    b_save_list_datasets(results,'compare_results_' + time.strftime('%Y%m%d%H',time.localtime(time.time())) + '.json')


def b_devide_data_import(data,task,method,threads):
    """
    标签标注数据集，并且划分为train和dev，最后上传到doccano，并且更新本地的数据库
    """

    b_save_df_datasets(data,'train_dev_imp.json')

    if method == 'process':
        b_label_dataset_multprocess(task,'train_dev_imp.json')
    else:
        b_label_dataset_mult(task,'train_dev_imp.json',threads)

    train_dev = b_read_dataset('train_dev_imp_label.json')

    train_dev = pd.DataFrame(train_dev)

    train,dev = b_split_train_test(train_dev,0.8)

    b_save_df_datasets(train,'train.json')
    b_save_df_datasets(dev,'dev.json')

    b_doccano_upload_by_task('train.json',task,'train')
    b_doccano_upload_by_task('dev.json',task,'dev')
    b_doccano_train_dev_update(task)

def b_check_duplicate_labels(file):
    """
    查询重复标签数据，打印出来
    """
    train_dev = b_read_dataset(file)

    for sample in train_dev:
        labels = sample['label']
        label_count = {}
        for label in labels:
            start = label[0]
            end = label[1]
            label_type = label[2]
            if label_type not in label_count:
                label_count[label_type] = 1
            else:
                label_count[label_type] += 1
    # 如果label_count中有一个label_type的数量大于1，则说明有重复的label
        for label_type in label_count:
            if label_count[label_type] > 1:
                print(sample['dataset'])
                print(sample['md5'])
                print(label_type + ':' + str(label_count[label_type]))
                print("-" * 10)


def b_check_overlap(file):
    """
    查找label是否overlap
    """
    train_dev = b_read_dataset(file)

    for sample in train_dev:
        labels = sample['label']
        label_count = {}
        for label in labels:
            start = label[0]
            end = label[1]
            label_type = label[2]
            for label_orig in labels:
                orig_start = label_orig[0]
                orig_end = label_orig[1]
                orig_type = label_orig[2]
                if label_type != orig_type:
                    if orig_end < start:
                        pass
                    elif orig_start > end:
                        pass
                    else:
                        print('-' * 20)
                        print(sample['dataset'])
                        print(sample['md5'])
                        print("{} {} {}".format(start, end, label_type))
                        print("{} {} {}".format(orig_start, orig_end, orig_type))

def b_select_data_by_mysql(task,label_name,num):
    """
    在数据库中，查找num个标注为label_name为空的数据，并且在doccano中没用的数据
    """
    labels = b_read_db_labels(task)
    label = labels[labels['label']==label_name]['col'].values[0]

    col = project_configs[task]['col']
    source = project_configs[task]['source']
    table = project_configs[task]['target']

    sql = 'select %s,%s from %s where %s is null' % (col,source,table,label)

    df = mysql_select_df(sql)

    df = p_process_df(df,task)

    db_dataset = b_read_db_datasets(task)
    df = df[df['md5'].isin(db_dataset['md5']) == False]
    df = df[:num]
    return df

def b_generate_cats_dataset_by_refine(org_data,compare_data,wrong_types):
    """
    根据refine的报表生成cats数据集
    """
    compare = b_read_dataset(compare_data)
    train_dev = b_read_dataset(org_data)

    md5s = []

    for entry in compare:
        if entry['wrong_type'] in wrong_types:
            md5s.append(entry['md5'])

    md5s = set(md5s)

    for sample in train_dev:
        sample['cats'] =  {"需要":0,"不需要":1} 
        if sample['md5'] in md5s:
            sample['cats'] =  {"需要":1,"不需要":0}

    p_generate_cats_datasets(train_dev)

def b_generate_metrics():
    """
    收集整理metrics
    """

    def record_metrics(task,date,item,metric,value):
        entry = {}
        entry['task'] = task
        entry['date'] = date
        entry['item'] = item
        entry['metric'] = metric
        entry['value'] = value
        new_data.append(entry)

    files = glob.glob('../training/metrics/*.json')
    new_data = []
    for file in files:
        file_name = file.split('/')[-1]
        task = file_name.split('_')[0]
        date = file_name.split('_')[1]
        entry = {}
        entry['task'] = task
        entry['date'] = date
        metric = json.load(open(file))
        record_metrics(task,date,'总体','p',metric['ents_p'])
        record_metrics(task,date,'总体','r',metric['ents_r'])
        record_metrics(task,date,'总体','f',metric['ents_f'])
        record_metrics(task,date,'速度','speed',metric['speed'])
        for key in metric['ents_per_type']:
            for key_2 in metric['ents_per_type'][key]:
                record_metrics(task,date,key,key_2,metric['ents_per_type'][key][key_2])

    df = pd.DataFrame(new_data)
    df.to_csv(ASSETS_PATH + 'metrics.csv',index=False)

def b_get_label_values(file):
    """
    找出所有label的value，并且保存到label_value.json文件中
    
    """
    data = b_read_dataset(file)

    new_data = []

    for sample in data:
        text = sample['data']
        labels = sample['label']
        for label in labels:
            entry = {'type':label[2],
                    'value':text[label[0]:label[1]],
                    'md5':sample['md5'],
                    'dataset':sample['dataset']}
            new_data.append(entry)

    b_save_list_datasets(new_data,'label_value.json')

def b_gpu_label(task,file):
    """
    服务器GPU直接标注
    """
    file_name = file.split('.')[0]
    data = b_read_dataset(file)

    nlp = b_load_best_model(task)

    data_data = [sample['data'] for sample in data]

    docs = nlp.pipe(data_data)

    for doc,sample in zip(docs,data):
        labels = []
        for ent in doc.ents:
            label = [ent.start_char,ent.end_char,ent.label_]
            labels.append(label)
        sample['label'] = labels

    b_save_list_datasets(data,file_name + '_label.json')

def b_combine_compare(files = ['1.xlsx','2.xlsx','3.xlsx']):
    """
    根据refine到处compare，合成compare文件
    """

    dfs = []

    for file in files:
        df = pd.read_excel(ASSETS_PATH + file)
        df.columns = [x.replace('_ - ', ' ').strip() for x in df.columns]
        dfs.append(df)

    df = pd.concat(dfs)

    b_generate_label_md5(df)

    df = df[df['is_human_correct'].str.upper() == 'Y']

    db = b_read_db_compare()
    # 合并
    df = pd.concat([db,df])
    
    df = df.drop_duplicates(subset=['label_md5'])

    b_save_db_compare(df)

    return df

def b_generate_label_md5(df):
    """
    设定排重列
    """
    # 设定排重列
    md5_columns = ['task','md5','human_start','human_end','human_label','ai_start','ai_end','ai_label','label_type']
    # md5 列组合生成md5值
    # 如果数字有.0，则去掉.0
    df['label_md5'] = df[md5_columns].apply(lambda x: ''.join(x.astype(str).apply(lambda x: x.replace('.0',''))),axis=1).apply(p_generate_md5)




def b_process_compare(file):
    """
    读取错题库并且处理
    """
    
    df = b_read_db_compare()

    compare = b_read_dataset(file)

    compare = pd.DataFrame(compare)

    b_generate_label_md5(compare)

    # 根据 label_md5 的值，将df中的is_human_correct的值填入到compare中
    compare['is_human_correct'] = compare['label_md5'].apply(lambda x: df[df['label_md5'] == x]['is_human_correct'].values[0] if x in df['label_md5'].values else None)
    
    b_save_df_datasets(compare,file)
# ——————————————————————————————————————————————————
# 调用
# ——————————————————————————————————————————————————

if __name__ == '__main__':
    pass
