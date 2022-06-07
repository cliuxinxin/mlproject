from re import X
from sqlalchemy import DDL
from tqdm import tqdm
from data_utils import *
from mysql_utils import *
import argparse
import glob
from data_clean_new import clean_manager


def change_unit(x):
    if '万' in str(x):
        return 10000
    else:
        return 1

def cal_amount(x,unit):
    return x * unit

def after_process(collector,task):
    """
    label之间的问题解决
    """
    if task == 'bid':
        try:
            collector['金额单位'] = collector['金额单位'].apply(lambda x: change_unit(x))
            collector['金额'] = collector['金额'].apply(lambda x: cal_amount(x,collector['金额单位']))
            return collector
        except:
            return collector
    if task == 'tender':
        try:
            collector['预算单位'] = collector['预算单位'].apply(lambda x: change_unit(x))
            collector['预算'] = collector.apply(lambda x: cal_amount(x['预算'],x['预算单位']),axis=1)
            return collector
        except:
            return collector



def is_sub(label_,task):
    """
    是否拆分子表
    """
    if task == 'tender':
        if label_ in ['招标单位']:
            return True
    if task == 'bid':
        if label_ in ['中标单位']:
            return True
    return False

def split_collector(collector,task):
    sub_collector = pd.DataFrame()
    for label in collector.columns:
        if is_sub(label,task):
            collector[label] = collector[label].apply(lambda x: x.split('#'))
            sub_collector = collector.explode(label)
            collector[label] = ''
    return sub_collector,collector

def process_df(task,df,std_labels):
    sub_collectors = []
    for i in range(len(df)):
        clean_labels = df.iloc[i]['clean_labels']
        if len(clean_labels) == 0:
            continue
        collector = {}
        for label in clean_labels:
            label_ = list(label.keys())[0]
            text = label[label_]
            if label_ not in collector:
                collector[label_] = text
            elif label_ in collector:
                if is_sub(label_,task):
                    collector[label_] = collector[label_] + '#' + text
        collector = pd.DataFrame(collector)
        collector = after_process(collector,task)
        sub_collector,collector = split_collector(collector,task)
        sub_collectors.append(sub_collector)
        labels = collector.columns.tolist()
        for label in labels:
            col_idx = std_labels[std_labels['label'] == label]['col_idx'].values[0]
            text = collector[label].values[0]
            df.iloc[i,col_idx] = text
    return sub_collectors
        


class Helper():
    def __init__(self) -> None:
        self.tender = b_load_best_model('tender')
        self.bid = b_load_best_model('bid')
        self.tender_label = pd.DataFrame(b_read_dataset('tender_train_dev.json'))
        self.bid_label = pd.DataFrame(b_read_dataset('bid_train_dev.json'))

    def get_model(self,task):
        if task == 'tender':
            return self.tender
        elif task == 'bid':
            return self.bid

    def get_label(self,task):
        if task == 'tender':
            return self.tender_label
        elif task == 'bid':
            return self.bid_label
        
def d_date_clean(value):
    """
    清洗日期
    """
    # 如果value是datetime.datetime类型，则直接返回
    if isinstance(value,datetime):
        if type(value) != pd._libs.tslibs.nattype.NaTType:
            return datetime.fromtimestamp(value.timestamp())

def datetime_process(df,task):
    """
    处理日期
    """
    if task == 'bid':
        datetime_columns = ['publish_time','publish_stime','publish_etime']
    elif task == 'tender':
        datetime_columns = ['publish_time','quote_stime','quote_etime','publish_stime','publish_etime']
    else:
        datetime_columns = []
    for colum in datetime_columns:
        # 转换为datetime格式
        try:
            df[colum] = df[colum].apply(lambda x: d_date_clean(x))
        except:
            pass
        # 转换为时间戳

    return df

def find_idx(cols,x):
    try:
        return cols.index(x)
    except:
        return None

def preprocess_df(df,task):
    """
    根据task做一些预处理
    """
    cols = df.columns.tolist()
    std_labels = b_read_db_labels(task)
    std_labels['col_idx'] = std_labels['col'].apply(lambda x: find_idx(cols,x))
    # 保留col_idx中有数的行
    std_labels = std_labels[std_labels['col_idx'].notnull()]
    html_col = 'detail_content'
    df['labels'] = ''
    df = datetime_process(df,task)
    return df,std_labels,html_col


def get_parser():
    parser = argparse.ArgumentParser(description="Process data and insert to mysql")
    return parser


def move_file(file):
    file_name = file.split('/')[-1]
    os.rename(file,DATA_PATH + 'processed/' + file_name)

def delete_mysql_by_df(target_table, df):
    ids = df['id'].to_list()
    if len(ids) == 1:
        id = ids[0]
        mysql_delete_data_by_id(id,target_table)
    else:
        mysql_delete_data_by_ids(ids,target_table)

def find_labels_by_md5(series,label_data):
    md5 = series['md5']
    try:
        labeled = label_data[label_data['md5'] == md5].iloc[0]['label']
    except:
        labeled = series['labels']
    return labeled

def process_labels(series,task):
    clean_labels = []
    for label in series['labels']:
        clean_label = {}
        label_ = list(label.keys())[0]
        text = label[label_]
        if label_ in std_labels['col'].to_list():
            col = std_labels[std_labels['label'] == label_].iloc[0]['col']
            clean_text = clean_manager(task,label_,text)
            clean_label[label_] = clean_text
            clean_labels.append(clean_label)
        else:
            clean_label[label_] = text
            clean_labels.append(clean_label) 
    return clean_labels

def save_labels(series,label):
    labels = series['clean_labels']
    save_labels = []
    for one_label in labels:
        label_ = list(one_label.keys())[0]
        text = one_label[label_]
        if label_ == label:
            save_labels.append(text)
    return "#".join(save_labels)

    
error = ASSETS_PATH + 'error.log'
# 删除error.log
if os.path.exists(error):
    os.remove(error)
    
files = glob.glob(DATA_PATH + '*.json')
helper = Helper()
file = files[0]

df = pd.read_json(file)

task = file.split('#')[0].split('/')[-1]
origin_table = file.split('#')[1]
target_table = b_get_target_table(origin_table)

label_data = helper.get_label(task)
nlp = helper.get_model(task)

df,std_labels,html_col = preprocess_df(df,task)
df['data'] = df[html_col].fillna('')
df['data'] = df['data'].apply(p_filter_tags)
df['md5'] = df['data'].apply(p_generate_md5)
max_len = 10000
df['data'] = df['data'].str[:max_len]

data = df['data'].to_list()

docs = nlp.pipe(data)

labels = []

for doc in docs:
    label = []
    for ent in doc.ents:
        label.append({ent.label_:ent.text})
    labels.append(label)

df['labels'] = labels

df['labels'] = df.apply(find_labels_by_md5,axis=1,args=(label_data,))

df['clean_labels'] = df.apply(process_labels,axis=1,args=(task,))

sub_collectors = process_df(task,df,std_labels)

df['labels'] = df['labels'].apply(lambda x: json.dumps(x,ensure_ascii=False))

df = df.drop(columns=['md5','data','clean_labels'])
delete_mysql_by_df(target_table, df)
mysql_insert_data(df,target_table)

