from redis_utils import sql_key,redis_
from mysql_utils import mysql_update
import json
import time


while True:
    while len(redis_.keys(sql_key)) > 0:
        data = redis_.lpop(sql_key)
        data = json.loads(data)
        try:
            mysql_update(data)
            print('Update one data')
        except:
            continue
    print('No sql to update')
    print("sleep 5s")
    time.sleep(5)

