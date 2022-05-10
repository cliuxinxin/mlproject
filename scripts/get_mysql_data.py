from data_utils import *
from mysql_utils import *
import time
import argparse
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description="Read data from mysql and save to json")
    parser.add_argument('--task', default='tender', help='task name')
    parser.add_argument('--mode', default='all', choices=['all', 'new'],help='all or newest')
    parser.add_argument('--number', default='100', help='save 100 records to a file')
    return parser

def get_all_data(task,table,number):
    sql = 'select count(1) from {} order by create_time desc '.format(table)
    df = mysql_select_df(sql)
    total = df.iloc[0][0]

    # 按照100条划分个数
    num = int(total / number)

    # 计算出每个文件的起始点
    start = 0
    end = number
    for i in range(num):
        start = end
        end = start + number
        sql = 'select * from {} order by create_time desc limit {} , {}'.format(table, start, number)
        df = mysql_select_df(sql)
        file_name = task + '_' + str(int(time.time()*100000))
        df.to_json(DATA_PATH + file_name + '.json')



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    task = args.task
    mode = args.mode
    number = int(args.number)
    table = project_configs[task]['table']
    if mode == 'all':
        get_all_data(task,table,number) 
    else:
        sql = "select * from %s  order by create_time desc limit %s" % (table,20)
    df = mysql_select_df(sql)
    for idx,i in tqdm(enumerate(range(0,len(df),number))):
        file_name = task + '_' + str(int(time.time()*100000))
        df[i:(i + number)].to_json(DATA_PATH + file_name + '.json')

    








