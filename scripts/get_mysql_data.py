from data_utils import *
from mysql_utils import *
import time
import argparse
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description="Read data from mysql and save to json")
    parser.add_argument('--task', default='bid', help='task name')
    parser.add_argument('--mode', default='new', choices=['all', 'new'],help='all or newest')
    parser.add_argument('--number', default='100', help='save 100 records to a file')
    return parser

def generate_sql(number, source, start,mode,max_time):
    if mode == 'all':
        sql = 'select * from {} order by create_time limit {} , {}'.format(source, start, number)
    if mode == 'new':
        sql = 'select * from {} where create_time > "{}" order by create_time limit {} , {}'.format(source, max_time, start, number)
    return sql

def get_data_divide_to_number(task, number, source, total,mode,max_time=''):
    # 按照100条划分个数
    num = int(total / number)

    # 计算出每个文件的起始点
    start = 0
    end = 0
    for i in tqdm(range(num)):
        start = end
        end = start + number
        sql = generate_sql(number, source, start,mode,max_time)
        df = mysql_select_df(sql)
        file_name = task + '_' + str(int(time.time()*100000))
        df.to_json(DATA_PATH + file_name + '.json')


def get_all_data(task,number):
    source = project_configs[task]['table']

    sql = 'select count(1) from {} order by create_time'.format(source)
    df = mysql_select_df(sql)
    total = df.iloc[0][0]
    print('total:',total)

    get_data_divide_to_number(task, number, source, total,mode='all')



def get_new_data(task,number):
    source = project_configs[task]['table']
    target = project_configs[task]['target']

    # 找到最新处理的数据
    sql = 'select create_time from {} order by create_time desc limit 1'.format(target)
    df = mysql_select_df(sql)
    max_time = df.iloc[0][0]
    print('max_time:',max_time)

    # 查找最新数据以后生成的数据
    sql = 'select count(1) from {} where create_time > "{}" order by create_time'.format(source, max_time)
    df = mysql_select_df(sql)
    total = df.iloc[0][0]
    print('total:',total)

    get_data_divide_to_number(task, number, source, total,mode='new',max_time=max_time)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    task = args.task
    mode = args.mode
    number = int(args.number)
    print('task:',task)
    print('mode:',mode)
    print('number:',number)
    if mode == 'all':
        get_all_data(task,number) 
    else:
        get_new_data(task,number)

    








