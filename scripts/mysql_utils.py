import pymysql
import configparser
import pandas as pd
from DBUtils.PooledDB import PooledDB



def d_parse_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

project_configs = d_parse_config()

def mysql_connect():
    """
    连接数据库
    """
    db = pymysql.connect(
        host=project_configs['mysql']['host'],
        port=int(project_configs['mysql']['port']),
        user=project_configs['mysql']['user'],
        passwd=project_configs['mysql']['password'],
        db=project_configs['mysql']['database'],
        charset='utf8'
    )
    return db

db = mysql_connect()

def mysql_select_df(sql):
    """
    执行sql，返回的数据转换成dataframe，并且表头是列名
    """
    import pandas as pd
    cursor = db.cursor()
    cursor.execute(sql)
    data = cursor.fetchall()
    df = pd.DataFrame(data)
    df.columns = [i[0] for i in cursor.description]
    return df

result_df = mysql_select_df('select * from tender_test limit 2')

def mysql_delete_data(df):
    """
    使用df的id，批量删除数据表中的数据
    """
    sql = 'delete from tender_test where id = %s'
    cursor = db.cursor()
    cursor.executemany(sql, df['id'].values.tolist())
    db.commit()

mysql_delete_data(result_df)

def mysql_insert_data(df):
    """
    使用df的表头和数据拼成批量更新的sql语句
    """
    sql = 'insert into tender_test ({}) values ({})'.format(','.join(df.columns), ','.join(['%s'] * len(df.columns)))
    cursor = db.cursor()
    values = df.values.tolist()
    # 讲NaT 替换成 ''
    for i in range(len(values)):
        for j in range(len(values[i])):
            if pd.isnull(values[i][j]):
                values[i][j] = None
    cursor.executemany(sql, values)
    db.commit()

mysql_insert_data(result_df)
















