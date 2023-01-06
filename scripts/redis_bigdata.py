import redis
from data_utils import project_configs
from mysql_utils import *
import time
from hdfs调用 import *
import json
import configparser
import ast

password = project_configs['REDIS_SLAVE1']['password']
host = project_configs['REDIS_SLAVE1']['host']
port = project_configs['REDIS_SLAVE1']['port']

# redis 参数设置
key1 = 'ods_path'
pool = redis.ConnectionPool(host=host, port=port, password=password, max_connections=50,db=13)
redis_ = redis.Redis(connection_pool=pool, decode_responses=True)
# # 获取所有数据
# keys=redis_.smembers(key1)
# print(keys)

# 目前能解析的表
start_list = ast.literal_eval(project_configs['data_process']['origin_tables'])
end_list = ast.literal_eval(project_configs['data_process']['target_tables'])
tasks = ast.literal_eval(project_configs['data_process']['tasks'])

# 取数据
def pop_redis_data(key):
    while True:
        try:
            s = redis_.spop(key)
            if isinstance(s,bytes):
                s = s.decode('utf-8')
        except:
            redis_.ping()
            time.sleep(5)
        else:
            break
    return s
# 日志
def logs(p):
    file = open('finished.log','a')
    file.write(str(p)+'\n')
    file.close()

if __name__ == "__main__":
    while True:
        try:
            path = pop_redis_data(key1)
            if path == None:
                logs("无可处理文件,程序终止。")
                break
            else:
                # path='/user/admin/ods/ZTB_data/test_other_tender_bid_result/dt=20230105/data013229_3.json'
                logs(f"正在处理文件：{path}")
                table = path.split('/')[5]
                if table in start_list:
                    origin_table = table
                    try: 
                        hdfs.read_hdfs_file(path,origin_table)
                    except Exception as e:
                        logs("Error1:"+str(e)+"**$$**"+path)
                        continue 
                else:
                    pass
        except Exception as e:
            logs("Error2:"+str(e)+"**$$**"+path)
            continue         
        time.sleep(10)