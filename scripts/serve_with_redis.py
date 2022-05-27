from data_utils import *
from mysql_utils import *
import time
import argparse
from tqdm import tqdm
import redis
import glob
from data_clean import clean_manager
from process_mysql_data import *
from get_mysql_data import *


# redis 参数设置
db_key = 'mywork'
pool = redis.ConnectionPool(host='localhost', port=6379, max_connections=50)
redis_ = redis.Redis(connection_pool=pool, decode_responses=True)

def get_parser():
    parser = argparse.ArgumentParser(description="Read data from mysql and save to json")
    return parser

def get_data():
    """
    读取所有的数据
    """
    process = b_get_dataprocess()
    for entry in process:
        task = entry['task']
        origin_table = entry['origin_table']
        target_table = entry['target_table']
        print('task:',task)
        print('origin_table:',origin_table)
        print('target_table:',target_table)
        get_diff_data(task,origin_table,target_table,number=100)
    files = glob.glob(DATA_PATH + '/*.json')
    for file in tqdm(files):
        redis_.rpush('task_export', file)
    print('get data done')

def parse_data(file):
    print(file)
    task = file.split('_')[0].split('/')[-1]
    label_data = b_read_dataset(task + '_train_dev.json')
    label_data = pd.DataFrame(label_data)
    df = pd.read_json(file)
    if df.empty:
        file_name = file.split('/')[-1]
        os.rename(file,DATA_PATH + 'processed/' + file_name)
        return 
    df,std_labels,html_col = preprocess_df(df,task)
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


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    number = int(args.number)
    while 1:
        redis_lenght = redis_.llen('task_export')
        print(f'There is {redis_lenght} files in redis')
        print('Start to parse data')
        if redis_lenght > 0:
            # 提取数据，数据即文件名
            file = redis_.lpop('task_export').decode('UTF-8')
            parse_data(file)
        time.sleep(60)
        print('Finish to parse data')
        print('Start to get data')
        get_data()

    








