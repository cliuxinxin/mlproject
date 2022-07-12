from data_utils import *
from mysql_utils import *
import time
import argparse
from tqdm import tqdm

def get_parser():
    """
    脚本文件参数解析
    """
    parser = argparse.ArgumentParser(description="Read data from mysql and save to json")
    parser.add_argument('--task', default='bid', help='task name')
    parser.add_argument('--mode', default='diff', choices=['all', 'new','diff'],help='all or newest')
    parser.add_argument('--number', default='200', help='save 100 records to a file')
    parser.add_argument('--table', default='', help='save 100 records to a file')
    return parser

def generate_sql(number, source, start,mode,max_time,target):
    """
    生成对应抽取sql
    """
    if mode == 'all':
        sql = 'select * from {} order by update_time limit {} , {}'.format(source, start, number)
    if mode == 'new':
        sql = 'select * from {} where update_time > "{}" order by update_time limit {} , {}'.format(source, max_time, start, number)
    if mode == 'diff':
        sql = 'select * from {} where  Not Exists (SELECT id from {} where {}.id = {}.id) order by update_time limit {} , {}'.format(source, target,target,source, start, number)
    return sql

def get_data_divide_to_number(task, number, source,total,mode,max_time='',target=''):
    """
    按照指定数量分割数据
    """
    num = int(total / number)

    # 计算出每个文件的起始点
    start = 0
    end = 0
    for i in tqdm(range(num)):
        start = end
        end = start + number
        sql = generate_sql(number, source, start,mode,max_time,target)
        df = mysql_select_df(sql)
        file_name = task + '#' + source + '#' + str(int(time.time()*100000))
        df.to_json(DATA_PATH + file_name + '.json')


def get_all_data(task,origin_table,number):
    """
    抽取所有数据
    """
    sql = 'select count(1) from {} order by update_time'.format(origin_table)
    df = mysql_select_df(sql)
    total = df.iloc[0][0]
    print('total:',total)

    get_data_divide_to_number(task, number, origin_table, total,mode='all')

def get_diff_data(task,origin_table,target_table,number):
    """
    抽取未处理数据
    """
    sql = 'select count(1) from {} where  Not Exists (SELECT id from {} where {}.id = {}.id) limit 100'.format(origin_table, target_table,target_table,origin_table)
    df = mysql_select_df(sql)
    total = df.iloc[0][0]
    print('total:',total)

    get_data_divide_to_number(task, number, origin_table, total,mode='diff',target=target_table)



def get_new_data(task,origin_table,target_table,number):
    """
    抽取最新数据
    """
    sql = 'select update_time from {} order by update_time desc limit 1'.format(target_table)
    df = mysql_select_df(sql)
    try:
        max_time = df.iloc[0][0]
    except:
        get_all_data(task,origin_table,number)
    print('max_time:',max_time)

    # 查找最新数据以后生成的数据
    sql = 'select count(1) from {} where update_time > "{}" order by update_time'.format(origin_table, max_time)
    df = mysql_select_df(sql)
    total = df.iloc[0][0]
    print('total:',total)

    get_data_divide_to_number(task, number, origin_table, total,mode='new',max_time=max_time)

def get_data(mode,task,origin_table,target_table,number):
    """
    抽取数据
    """
    if mode == 'all':
        print('get all data')
        print('origin_table:',origin_table)
        get_all_data(task,origin_table,number)
    if mode == 'new':
        print('get new data')
        print('origin_table:',origin_table)
        print('target_table:',target_table) 
        get_new_data(task,origin_table,target_table,number)
    if mode == 'diff':
        print('get diff data')
        print('origin_table:',origin_table)
        print('target_table:',target_table)
        get_diff_data(task,origin_table,target_table,number)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    task = args.task
    mode = args.mode
    number = int(args.number)
    table = args.table
    process = b_get_dataprocess()
    print('task:',task)
    print('mode:',mode)
    print('number:',number)
    print('table:',table)
    for entry in process:
        origin_table = entry['origin_table']
        target_table = entry['target_table']
        if table != '' and origin_table == table:
            get_data(mode,task,origin_table,target_table,number)
            break
        if table == '':
            get_data(mode,task,origin_table,target_table,number)