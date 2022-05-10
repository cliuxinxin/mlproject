from data_utils import *
from mysql_utils import *
import time
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Read data from mysql and save to json")
    parser.add_argument('--task', default='tender', help='task name')
    parser.add_argument('--mode', default='all', choices=['all', 'new'],help='all or newest')
    parser.add_argument('--number', default='100', help='save 100 records to a file')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    task = args.task
    mode = args.mode
    number = int(args.number)
    table = project_configs[task]['table']
    if mode == 'all':
        sql = 'select * from {} order by create_time desc '.format(table)
    else:
        sql = "select * from %s  order by create_time desc limit %s" % (table,100)
    df = mysql_select_df(sql)
    for idx,i in enumerate(range(0,len(df),number)):
        file_name = task + '_' + str(int(time.time()*100000))
        df[i:(i + number)].to_json(DATA_PATH + file_name + '.json')
    








