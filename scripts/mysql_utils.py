import pymysql
import configparser
import pandas as pd
from dbutils.persistent_db import PersistentDB

MYSQL = 'mysql-test'

def d_parse_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

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


def mysql_select_df(sql):
    """
    执行sql，返回的数据转换成dataframe，并且表头是列名
    """
    import pandas as pd
    cursor = conn.cursor()
    cursor.execute(sql)
    data = cursor.fetchall()
    df = pd.DataFrame(data)
    df.columns = [i[0] for i in cursor.description]
    return df

def mysql_delete_data(df):
    """
    使用df的id，批量删除数据表中的数据
    """
    sql = 'delete from tender_test where id = %s'
    cursor = conn.cursor()
    cursor.executemany(sql, df['id'].values.tolist())
    db.commit()


def mysql_insert_data(df):
    """
    使用df的表头和数据拼成批量更新的sql语句
    """
    sql = 'insert into tender_test ({}) values ({})'.format(','.join(df.columns), ','.join(['%s'] * len(df.columns)))
    cursor = conn.cursor()
    values = df.values.tolist()
    # 将NaT 替换成 ''
    for i in range(len(values)):
        for j in range(len(values[i])):
            if pd.isnull(values[i][j]):
                values[i][j] = None
    cursor.executemany(sql, values)
    db.commit()

















