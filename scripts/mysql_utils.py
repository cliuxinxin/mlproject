import pymysql
import configparser


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

result_df = mysql_select_df('select * from tender_test limit 1')

# 根据id更新liaison_pnumber的数据
def mysql_update_liaison_pnumber(id, pnumber):
    sql = 'update tender_test set liaison_pnumber = "{}" where id = {}'.format(pnumber, id)
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()






