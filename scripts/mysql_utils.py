import pymysql
import configparser
import pandas as pd


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

def mysql_delete_data(df):
    """
    读取df的id字段，并且将数据删除
    """
    for id in df['id']:
        sql = "delete from tender_test where id = '{}'".format(id)
        cursor = db.cursor()
        cursor.execute(sql)
        db.commit()

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
















