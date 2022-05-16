import pymysql
import configparser
import pandas as pd
from dbutils.persistent_db import PersistentDB
import os
from threading import RLock

MYSQL = 'mysql'

def d_parse_config():
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
    return config

def d_log_error(sql):
    """
    打印错误日志
    """
    with open('..assets/error.log','a+') as f:
        f.write(sql)

project_configs = d_parse_config()

def mysql_connect(config='mysql'):
    """
    连接数据库
    """
    db = pymysql.connect(
                        host=project_configs[config]['host'],
                        port=int(project_configs[config]['port']),
                        user=project_configs[config]['user'],
                        passwd=project_configs[config]['password'],
                        db=project_configs[config]['database'], 
                        charset='utf8'
    )
    return db

db = mysql_connect(MYSQL)

def mysql_connect_pool(config='mysql'):
    pool = PersistentDB(
                    pymysql,
                    5,
                    host=project_configs[config]['host'],
                    port=int(project_configs[config]['port']),
                    user=project_configs[config]['user'],
                    passwd=project_configs[config]['password'],
                    db=project_configs[config]['database'],
                    charset='utf8',
                    setsession=['SET AUTOCOMMIT = 1']
                    )
                        
    return pool


pool = mysql_connect_pool(MYSQL)

conn = pool.connection()

LOCK = RLock()

def mysql_select_df(sql):
    """
    执行sql，返回的数据转换成dataframe，并且表头是列名
    """
    import pandas as pd
    with LOCK:
        cursor = conn.cursor()  
        cursor.execute(sql)
        data = cursor.fetchall()
        cursor.close()
    df = pd.DataFrame(data)
    df.columns = [i[0] for i in cursor.description]
    return df

def mysql_select_data_by_ids(ids,table):
    """
    使用id批量查询数据
    """
    sql = "select * from {} where id in {}".format(table,tuple(ids))
    df = mysql_select_df(sql)
    return df

def mysql_delete_data(df,task):
    """
    使用df的id，批量删除数据表中的数据
    """
    sql = 'delete from %s where id in %s'
    cursor = conn.cursor()
    cursor.executemany(sql,(project_configs[task]['target'],df['id'].values.tolist()))
    db.commit()


def mysql_delete_data_by_id(id,task):
    """
    使用id删除数据表中的数据
    """
    sql = "delete from {} where id = '{}'".format(project_configs[task]['target'],id)
    with LOCK:
        cursor = conn.cursor()
        cursor.execute(sql)
        db.commit()
        cursor.close()


def mysql_insert_data(df,task):
    """
    使用df的表头和数据拼成批量更新的sql语句
    """
    sql = 'insert into {} ({}) values ({})'.format(project_configs[task]['target'],','.join(df.columns), ','.join(['%s'] * len(df.columns)))
    
    values = df.values.tolist()
    # 将NaT 替换成 ''
    for i in range(len(values)):
        for j in range(len(values[i])):
            if pd.isnull(values[i][j]):
                values[i][j] = None
    with LOCK:
        cursor = conn.cursor()
        try:
            cursor.executemany(sql, values)
        except:
            for value in values:
                try:
                    cursor.execute(sql, value)
                except:
                    d_log_error(sql)
        db.commit()
        cursor.close()

def mysql_delete_data_by_ids(ids,task):
    """
    使用ids删除对应的表格中的数据
    """
    sql = "delete from {} where id in {}".format(project_configs[task]['target'],tuple(ids))
    with LOCK:
        cursor = conn.cursor()
        cursor.execute(sql)
        db.commit()
        cursor.close()

















