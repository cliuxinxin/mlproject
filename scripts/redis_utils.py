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
tag_key = 'tag'
sql_key = 'sql'
pool = redis.ConnectionPool(host=host, port=port, password=password, max_connections=50)
redis_ = redis.Redis(connection_pool=pool, decode_responses=True)

process_config_bussiness  = ['tender','bid','contract']

def read_config():
    with open('data_process.json', 'r') as f: #,encoding='utf-8'
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

def get_tag_config(configs):
    new_configs = {}
    for key,value in configs.items():
        new_configs[value['target']] = value['task'] + 'cats'
    return new_configs
            

def redis_push(df,key):
    for idx,row in df.iterrows():
        content = json.dumps(row.to_dict())
        redis_.rpush(key,content)

def redis_push_diff(ori_tar_configs):
    for ori,tar in ori_tar_configs.items():
    # 读取差异总数据
        sql = f'select id from {ori} where id not in (select id from {tar})'
        # 取100条数据
        # sql = f'select id from {ori} limit 100'

        df = mysql_select_df(sql)
        df['table'] = ori

        redis_push(df,diff_key)

def redis_push_tag(tag_configs):
    for table,task in tag_configs.items():
        sql = f"select id from {table} where classify_type is null and is_full_data = 1"
        df = mysql_select_df(sql)
        df['table'] = table
        df['task'] = task
        redis_push(df,tag_key)

def redis_push_tag_test(tag_configs):
    for table,task in tag_configs.items():
        sql = f"select id from {table} where classify_type is null limit 200"
        df = mysql_select_df(sql)
        df['table'] = table
        df['task'] = task
        redis_push(df,tag_key)

def redis_push_all_tender():
    for ori in ['test_tender_bid']:   #,'test_other_tender_bid'
        # 读取所有id数据
        sql = f'select id from {ori}'
        df = mysql_select_df(sql)
        df['table'] = ori

    redis_push(df,diff_key)

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

def generate_file(ori_task_configs,num=100):
    data = pop_redis_data(diff_key,num)
    save_data(data,ori_task_configs)


process_configs = read_config()
ori_tar_configs = get_target_config(process_configs)
ori_task_configs = get_task_config(process_configs)
tag_configs = get_tag_config(process_configs)



# redis_push_tag_test(tag_configs)

# redis_push_diff(ori_tar_configs
# generate_file(ori_task_configs,num=100)

# # 打印出redis的长度
# print(redis_.llen(diff_key))
# print(redis_.llen(tag_key))

# # 删除redis中的数据
# redis_.delete(tag_key)



