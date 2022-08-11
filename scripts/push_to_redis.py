import redis
from data_utils import project_configs,DATA_PATH
from mysql_utils import *
import json
import time

password = project_configs['redis']['password']
host = project_configs['redis']['host']
port = project_configs['redis']['port']

# redis 参数设置
diff_key = 'diff'
file_key = 'file'
pool = redis.ConnectionPool(host=host, port=port, password=password, max_connections=50)
redis_ = redis.Redis(connection_pool=pool, decode_responses=True)

process_config_bussiness  = ['tender','bid']

def read_config():
    with open('data_process.json', 'r') as f:
        configs = json.load(f)

    for item in process_config_bussiness:
        del configs[item]

    return configs

def parse_configs(configs,keyword):
    new_configs = {}
    for key,value in configs.items():
        for key2,value2 in value.items():
            if key2 == keyword:
                new_configs[key] = value2
    return new_configs

def get_target_config(configs):
    return parse_configs(configs,'target')

def get_task_config(configs):
    return parse_configs(configs,'task')

def redis_push_diff(ori_tar_configs):
    for ori,tar in ori_tar_configs.items():
    # 读取差异总数据
        # sql = f'select id from {ori} where id not in (select id from {tar}) limit 100'
        # 取100条数据
        sql = f'select id from {ori} limit 100'

        df = mysql_select_df(sql)
        df['table'] = ori

    # 将df的数据推送到redis
        for idx,row in df.iterrows():
            content = json.dumps(row.to_dict())
            # 如果不存在，则推送
            if not redis_.hexists(diff_key,content):
                redis_.rpush(diff_key,content)

def redis_not_empty(key):
    return redis_.llen(key) > 0

def pop_redis_data(key,num):
    data = []
    for i in range(num):
        try:
            s = redis_.lpop(key)
            data.append(json.loads(s))
        except:
            break
    return data

def save_data(data,ori_task_configs):
    df = pd.DataFrame(data)
    df_group = df.groupby('table')
    for ori,idxs in df_group.groups.items():
        sql = f"select * from {ori} where id in {tuple(df.loc[df_group.groups[ori]].id.values.tolist())}"
        df1 = mysql_select_df(sql)
        task = ori_task_configs[ori]
        file_name = task + '#' + ori + '#' + str(int(time.time()*100000))
        df1.to_json(DATA_PATH + file_name + '.json')
        print(len(df1))
        redis_.rpush(file_key, file_name + '.json')

def generate_file(ori_task_configs,num=100):
    while redis_not_empty(diff_key):
        data = pop_redis_data(diff_key,num)
        save_data(data,ori_task_configs)

process_configs = read_config()
ori_tar_configs = get_target_config(process_configs)
ori_task_configs = get_task_config(process_configs)

redis_push_diff(ori_tar_configs)
generate_file(ori_task_configs,num=100)

# 打印出redis的长度
print(redis_.llen(diff_key))

