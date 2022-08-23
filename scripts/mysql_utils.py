import pymysql
import configparser
import pandas as pd
from dbutils.persistent_db import PersistentDB
import os
from threading import RLock
import json
import copy

MYSQL = 'mysql'
test = False

def d_parse_config():
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
    return config

def d_log_error(error):
    """
    打印错误日志
    """
    with open('../assets/error.log','a+',encoding='utf-8') as f:
        for key,value in error.items():
            if key == 'error':
                f.write(str(value))
                f.write('#')
            if key == 'value':
                f.write(str(value))
                f.write('#')
            if key == 'table':
                f.write(str(value))
        f.write('\n')
        
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
    try:
        df.columns = [i[0] for i in cursor.description]
    except:
        # 为空
        pass
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


def mysql_delete_data_by_id(id,table):
    """
    使用id删除数据表中的数据
    """
    sql = "delete from {} where id = '{}'".format(table,id)
    with LOCK:
        cursor = conn.cursor()
        cursor.execute(sql)
        db.commit()
        cursor.close()

def mysql_delete_win_by_id(id,table_name,table):
    """
    使用id删除数据表中的数据
    """
    sql  = f"delete from {table} where table_name = '{table_name}' and announcement_id = '{id}'"
    with LOCK:
        cursor = conn.cursor()
        cursor.execute(sql)
        db.commit()
        cursor.close()

def mysql_delete_win_by_ids(ids,table_name,table):
    """
    使用ids删除对应的表格中的数据
    """
    sql = f"delete from {table} where table_name = '{table_name}' and announcement_id in {tuple(ids)}"
    with LOCK:
        cursor = conn.cursor()
        cursor.execute(sql)
        db.commit()
        cursor.close()

def mysql_insert_data(df,table):
    """
    使用df的表头和数据拼成批量更新的sql语句
    """
    sql = 'insert into {} ({}) values ({})'.format(table,','.join(df.columns), ','.join(['%s'] * len(df.columns)))
    
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
                if test:
                    cursor.execute(sql, value) 
                else:
                    try:
                        cursor.execute(sql, value)
                    except Exception as e:
                        error = {'error':e,'value':value[0],'table':table}
                        d_log_error(error)
        db.commit()
        cursor.close()

def mysql_delete_data_by_ids(ids,table):
    """
    使用ids删除对应的表格中的数据
    """
    sql = "delete from {} where id in {}".format(table,tuple(ids))
    with LOCK:
        cursor = conn.cursor()
        cursor.execute(sql)
        db.commit()
        cursor.close()

def mysql_update(dict_value):
    """
    使用insert into语句批量更新表格的数据
    """
    tmp  = copy.deepcopy(dict_value)
    table = tmp.pop('table')
    id = tmp.pop('id')
    partial_sql = ','.join([f"{key} = '{value}'" for key,value in tmp.items()])
    sql = f"update {table} set {partial_sql} where id = '{id}'"
    with LOCK:
        cursor = conn.cursor()
        cursor.execute(sql)
        db.commit()
        cursor.close()















