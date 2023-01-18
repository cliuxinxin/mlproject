from redis_utils import generate_file,ori_task_configs,redis_,diff_key
from data_utils import DATA_PATH
import os
import time

# 读取文件目录下面的json文件列表
def read_json_files(path):
    files = os.listdir(path)
    return [file for file in files if file.endswith('.json')]
    
def is_file_not_enough(files,num):
    return len(files) < num

# 判断redis中diff_key数据是否为空
def is_redis_not_empty(redis_,key):
    return redis_.llen(key) > 1

files = read_json_files(DATA_PATH)
# 如果文件不满8个，并且redis不为空,则生成文件
while True:
    if is_file_not_enough(files,8) and is_redis_not_empty(redis_,diff_key):
        print(f'There is {redis_.llen(diff_key)} data in redis')
        try:
            generate_file(ori_task_configs,200)
            print('Generate file done')
        except:
            continue
    files = read_json_files(DATA_PATH)
    time.sleep(5)