from tqdm import tqdm
from data_utils import *
from mysql_utils import *
import argparse
import glob
from data_clean_new import clean_manager



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

def move_file(file):
    """
    移动已处理文件
    """
    file_name = file.split('/')[-1]
    os.rename(file,DATA_PATH + 'processed/' + file_name)

def delete_win_by_df(target_table, df, origin_table):
    """
    删除数据
    """
    ids = df['announcement_id'].to_list()
    if len(ids) == 1:
        id = ids[0]
        mysql_delete_win_by_id(id,origin_table,target_table)
    else:
        mysql_delete_win_by_ids(ids,origin_table,target_table)

def delete_mysql_by_df(target_table, df):
    """
    删除数据
    """
    ids = df['id'].to_list()
    if len(ids) == 1:
        id = ids[0]
        mysql_delete_data_by_id(id,target_table)
    else:
        mysql_delete_data_by_ids(ids,target_table)

def find_labels_by_md5(md5,label_data):
    """
    根据md5查找标签，填充到label里面
    """
    try:
        labels = label_data.loc[md5]['labels']
    except:
        labels = []
    return labels

def get_labels(series):
    """
    整理标签
    """
    text = series['data']
    new_labels = []
    labels = series['labels']
    if len(labels) == 0:
        return new_labels
    for start,end,label_ in labels:
        new_label = {}
        new_label[label_] = text[start:end]
        new_labels.append(new_label)
    return new_labels

def clean_labels(labels,task):
    """
    根据任务和label名称清洗label
    """
    clean_labels = []
    for label in labels:
        clean_label = {}
        label_,text = list(label.items())[0]
        clean_text = clean_manager(task,label_,text) 
        clean_label[label_] = clean_text  
        clean_labels.append(clean_label)
    return clean_labels

def process_labels(labels,task):
    """
    根据任务做一些label之间的数据处理
    """
    if len(labels) == 0:
        return labels
    if task == 'bid':
        # 找到labels中的中标金额和中标金额单位
        unit = 1
        for label in labels:
            label_,text = list(label.items())[0]
            if label_ == '中标金额':
                amount = text
            if label_ == '中标金额单位':
                unit = text
        # 把中标金额和中标金额单位拼接起来
        new_labels = []
        for label in labels:
            new_label = {}
            label_,text = list(label.items())[0]
            new_label[label_] = text
            if label_ == '中标金额':
                new_label[label_] = float(amount) * float(unit)
            new_labels.append(new_label)
        return new_labels
    return labels

def preprocess_save(labels,task):
    """
    根据任务做多个labels的处理。
    """
    new_labels = {}
    if len(labels) == 0:
        return {}
    mult_labels = []  
    if task == 'bid':
        mult_labels = ['中标单位']
    for label in labels:
        label_,text = list(label.items())[0]
        if label_ in mult_labels and label_ in new_labels:
            new_labels[label_] = new_labels[label_] + '#' + text
        else:
            new_labels[label_] = text
    return new_labels

def fill_data(x,label_df,col):
    """
    填充数据
    """
    try:
        result = label_df[label_df['md5']==x][col].values[0]
    except:
        result = ''
    return result

def process_save(data_process, task, origin_table, df):
    """
    处理数据
    """
    data = []
    sub_data = []

    for md5,labels,id in df[['md5','clean_labels','id']].values:
        if len(labels) == 0:
            continue
        df_labels = pd.DataFrame([labels])
        colunms = list(df_labels.columns)
        new_columns = []
        for column in colunms:
            col = data_process.get(task).get(column)
            if col:
                new_columns.append(col)
            else:
                new_columns.append(column)
        df_labels.columns = new_columns
        df_labels['md5'] = md5
        if task == 'bid':
            if 'winning_bidder' in df_labels.columns:
                df_labels['winning_bidder'] = df_labels['winning_bidder'].apply(lambda x: x.split('#'))
                split_df_labels = df_labels.explode('winning_bidder')
                split_df_labels[['announcement_id','table_name']] = [id,origin_table]
                if 'amount' in split_df_labels.columns and len(split_df_labels) > 1:
                # 保留第一行amount,其他行清空
                    split_df_labels[1:,'amount'] = 0
                sub_data.append(split_df_labels)
            df_labels[['winning_bidder','amount']] = ''
        data.append(df_labels)
        
    label_df = pd.concat(data)
    if task == 'bid':
        sub_label_df = pd.concat(sub_data)
        if 'amount' not in sub_label_df.columns:
            sub_label_df['amount'] = 0
        sub_label_df = sub_label_df[['winning_bidder','amount','table_name','announcement_id']]
    else:
        sub_label_df = pd.DataFrame()
    return label_df,sub_label_df

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
            bid_label = self.bid_label[['md5','label']]
            bid_label.columns = ['md5','labels']
            bid_label.set_index('md5',inplace=True)
            return bid_label
    
   
if __name__ == '__main__':
    files = glob.glob(DATA_PATH + '*.json')
    helper = Helper()
    data_process = json.loads(open('data_process.json','r',encoding='utf-8').read())

    for file in tqdm(files):
        print(file)
        task = file.split('#')[0].split('/')[-1]
        origin_table = file.split('#')[1]
        target_table = data_process.get(origin_table).get('target')

        label_data = helper.get_label(task)
        nlp = helper.get_model(task)

        df = pd.read_json(file)
        df['data'] = df['detail_content'].fillna('')
        df['data'] = df['data'].apply(p_filter_tags)
        df['md5'] = df['data'].apply(p_generate_md5)
        max_len = 10000
        df['data'] = df['data'].str[:max_len]
        # 计算标签
        data = df['data'].to_list()
        docs = nlp.pipe(data)
        labels = []

        for doc in docs:
            label = []
            for ent in doc.ents:
                label.append([ent.start_char,ent.end_char,ent.label_])
            labels.append(label)

        df['labels'] = labels
        # 替换正确标签
        df[df.md5.isin(label_data.index)]['labels'] = df['md5'].apply(lambda x:find_labels_by_md5(x,label_data))
        # 得到label内容
        df['labels'] = df.apply(get_labels,axis=1)
        # label根据任务做清洗
        df['clean_labels'] = df['labels'].apply(lambda x:clean_labels(x,task))
        # 根据任务，处理label之间的问题
        df['clean_labels'] = df['clean_labels'].apply(lambda x:process_labels(x,task))
        # 根据任务预处理数据，合并标签内容，有全部合并成一个或者是取最前面的数据
        df['clean_labels'] = df['clean_labels'].apply(lambda x:preprocess_save(x,task))
        # 处理填充数据，可能根据任务分列
        label_df, sub_label_df = process_save(data_process, task, origin_table, df)
        # 填充label数据
        cols = list(data_process[task].values())
        for col in label_df.columns:
            for md5,value in label_df[label_df[col].notnull()][['md5',col]].values:
                df.loc[df.md5 == md5,col] = value
        # 填充子表
        if task == 'bid':
            delete_win_by_df('final_winning_bidder',sub_label_df,origin_table)
            mysql_insert_data(sub_label_df,'final_winning_bidder')

        # 清理df数据
        df['labels'] = df['labels'].apply(lambda x: json.dumps(x,ensure_ascii=False))
        df = df.drop(columns=['md5','data','clean_labels'])
        df = datetime_process(df,task)

        # 填写数据
        delete_mysql_by_df(target_table, df)
        mysql_insert_data(df,target_table)
        move_file(file)
