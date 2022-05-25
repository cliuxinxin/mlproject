
from tqdm import tqdm
from data_utils import *
from mysql_utils import *
import argparse
import glob
from data_clean import clean_manager

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
         df[colum] = df[colum].apply(lambda x: d_date_clean(x))
        # 转换为时间戳

    return df

def preprocess_df(df,task):
    """
    根据task做一些预处理
    """
    cols = df.columns.tolist()
    std_labels = b_read_db_labels(task)
    std_labels['col_idx'] = std_labels['col'].apply(lambda x: cols.index(x))
    html_col = project_configs[task]['col']
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


class PoolCorpus(object):

    def __init__(self,task,file):
        df = pd.read_json(file)
        self.task = task
        df,std_labels,html_col = preprocess_df(df,task)
        self.html_col = html_col
        self.df = df
        self.std_labels = std_labels
        self.nlp = b_load_best_model(task)

    def add(self, i):
        process_df(i,self.df,self.html_col,self.nlp,self.std_labels,self.task) 


    def get(self):
        return self.df


def work(q,df,html_col,nlp,std_labels,task):
        while True:
            if q.empty():
                return
            else:
                idx = q.get()
                process_df(idx,df,html_col,nlp,std_labels,task)
                

def get_parser():
    parser = argparse.ArgumentParser(description="Process data and insert to mysql")
    parser.add_argument('--mode', default='pipe', choices=['process', 'thread','single','pipe'],help='all or newest')
    parser.add_argument('--thread_num', default=5 ,help='if mode is thread, set thread num')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    mode = args.mode
    thread_num = int(args.thread_num)
    files = glob.glob(DATA_PATH + '*.json')

    BaseManager.register('PoolCorpus', PoolCorpus)

    for file in tqdm(files):
        print(file)
        task = file.split('#')[0].split('/')[-1]
        label_data = b_read_dataset(task + '_train_dev.json')
        label_data = pd.DataFrame(label_data)
        if mode == 'process':
            with BaseManager() as manager:
                corpus = manager.PoolCorpus(task,file)

                with Pool() as pool:
                    pool.map(corpus.add, (i for i in range(len(corpus.get()))))

                df = corpus.get()
        elif mode == 'thread':
            df = pd.read_json(file)
            df,std_labels,html_col = preprocess_df(df,task) 
            nlp = b_load_best_model(task)
            q = queue.Queue()
            for idx in range(len(df)):
                q.put(idx)
            thread_num = thread_num
            threads = []
            for j in range(thread_num):
                t = threading.Thread(target=work, args=(q,df,html_col,nlp,std_labels,task,label_data))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
        elif mode == 'single':
            df = pd.read_json(file)
            df,std_labels,html_col = preprocess_df(df,task) 
            nlp = b_load_best_model(task)
            for idx in range(len(df)):
                process_df(idx,df,html_col,nlp,std_labels,task,label_data)

        else:
            df = pd.read_json(file)
            df,std_labels,html_col = preprocess_df(df,task)
            # 将为none的替换成空
            df[html_col] = df[html_col].fillna('')
            df[html_col] = df[html_col].apply(p_filter_tags)
            data = df[html_col].to_list()
            nlp = b_load_best_model(task)
            docs = nlp.pipe(data)
            for idx,doc in enumerate(docs):
                process_df(idx,df,html_col,nlp,std_labels,task,label_data,doc)
        
        ids = df['id'].to_list()
        if len(ids) == 1:
            id = ids[0]
            mysql_delete_data_by_id(id,task)
        else:
            mysql_delete_data_by_ids(ids,task)
        mysql_insert_data(df,task)
        file_name = file.split('/')[-1]
        os.rename(file,DATA_PATH + 'processed/' + file_name)

