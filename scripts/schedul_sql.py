from redis_utils import sql_key,redis_
from mysql_utils import mysql_update
import json

s = redis_.lpop(sql_key)
s = json.loads(s)

mysql_update(s)