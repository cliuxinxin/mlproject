
from tqdm import tqdm
from data_utils import *
from mysql_utils import *
import argparse
import glob
from data_clean import clean_manager

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
        datetime_columns = ['publish_time','quote_stime','quote_etime','publish_stime','publish_etime','tender_etime']
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

def process_df(i,df,html_col,nlp,std_labels,task,label_data,doc=None):
    """
    处理数据
    """
    text = df.iloc[i][html_col]
    md5 = p_generate_md5(text)
    try:
        labeled = label_data[label_data['md5'] == md5].iloc[0]['label']
    except:
        labeled = []
    if text is None:
        return
    if doc == None:
        text = p_filter_tags(text)
        doc = nlp(text)
    labels = []
    if labeled != []:
        for single_label in labeled:
            start = single_label[0]
            end = single_label[1]
            label_ = single_label[2]
            label = {}
            label_text = text[start:end]
            label[label_] = label_text
            if label_ in std_labels['label'].to_list():
                col_idx = std_labels[std_labels['label'] == label_].iloc[0]['col_idx']
                col = std_labels[std_labels['label'] == label_].iloc[0]['col']
                clean_label = clean_manager(task,col,label_text)
                df.iloc[i,col_idx] = clean_label
            labels.append(label)
        df.iloc[i,-1] = json.dumps(labels,ensure_ascii=False)   
        return
    for ent in doc.ents:
        label = {}
        label[ent.label_] = ent.text.strip()
        if ent.label_ in std_labels['label'].to_list():
            col_idx = std_labels[std_labels['label'] == ent.label_].iloc[0]['col_idx']
            col = std_labels[std_labels['label'] == ent.label_].iloc[0]['col'] 
            clean_label = clean_manager(task,col,ent.text)
            df.iloc[i,col_idx] = clean_label
        labels.append(label)        
    df.iloc[i,-1] = json.dumps(labels,ensure_ascii=False)  

def get_parser():
    parser = argparse.ArgumentParser(description="Process data and insert to mysql")
    parser.add_argument('--test', default='N', choices=['Y','N'],help='test')
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

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    test = args.test
    files = glob.glob(DATA_PATH + '*.json')
    helper = Helper()

    for file in tqdm(files):
        print(file)

        df = pd.read_json(file)
        if df.empty:
            move_file(file)
            continue 
        if test == 'Y':
            df = df[:5]

        task = file.split('#')[0].split('/')[-1]
        origin_table = file.split('#')[1]
        target_table = b_get_target_table(origin_table)

        label_data = helper.get_label(task)
        nlp = helper.get_model(task)

        df,std_labels,html_col = preprocess_df(df,task)
        df[html_col] = df[html_col].fillna('')
        df[html_col] = df[html_col].apply(p_filter_tags)
        # 限制html_col最长为 1400000
        max_len = 50000
        df[html_col] = df[html_col].str[:max_len]
        data = df[html_col].to_list()
        docs = nlp.pipe(data)

        for idx,doc in enumerate(docs):
            process_df(idx,df,html_col,nlp,std_labels,task,label_data,doc)
        
        delete_mysql_by_df(target_table, df)
        mysql_insert_data(df,target_table)
        move_file(file)

