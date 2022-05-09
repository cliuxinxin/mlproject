from data_utils import *
from mysql_utils import *
import time
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Read data from mysql and save to json")
    parser.add_argument('--task', default='tender', help='task name')

    
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    task = args.task
    print('Hello {}'.format(task))