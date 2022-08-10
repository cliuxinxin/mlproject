import pymysql
import configparser
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

# MYSQL = 'mysql'
test = True

def d_parse_config():
    config = configparser.ConfigParser()
    config.read("config.ini")
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
conn = mysql_connect()
cur = conn.cursor()


tables = {
    "test_other_tender_bid_result":{
        "target": "final_other_tender_bid_result",
        "task": "bid"
    },
    "test_other_tender_bid": {
        "target": "final_other_tender_bid", 
        "task": "tender"
    },
    "test_procurement_bid": {
        "target": "final_procurement_bid", 
        "task": "tender"
    },
    "test_procurement_bid_result": {
        "target": "final_procurement_bid_result", 
        "task": "bid"
    },
    "test_tender_bid": {
        "target": "final_tender_bid", 
        "task": "tender"
    },
    "test_tender_bid_result": {
        "target": "final_tender_bid_result", 
        "task": "bid"
    }
}


def get_count(table):
    cur.execute(f'select count(*) from {table}')
    count = cur.fetchone()
    return count[0]


# 查询a表中不在b表的数据数量
def get_count_a_not_b(table_a,table_b):
    cur.execute(f'select count(*) from {table_a} where id not in (select id from {table_b})')
    count = cur.fetchone()
    return count[0]
tender_table = []
tender_count = []
bid_table = []
bid_count = []
datas = [
    [[],[]],
    [[],[]]
]
for table in tables.keys():
    if tables[table]['task'] == 'tender':
        datas[0][0].append([table,get_count(table)])
        datas[0][1].append([tables[table]['target'],get_count(tables[table]['target'])])
    if tables[table]['task'] == 'bid':
        datas[1][0].append([table,get_count(table)])
        datas[1][1].append([tables[table]['target'],get_count(tables[table]['target'])])
    # print("任务：{tables[table]['task']}\n{table}表中有{get_count(table)}条数据")
    # print("任务：{tables[table]['task']}\n{0}表中有{1}条数据".format(tables[table]['target'],get_count(tables[table]['target'])))
cur.close()
conn.close()
data = copy.deepcopy(datas)

# 增加分类的总数
data[0][0].append(['tender_test_total',np.sum(np.array(data[0][0])[:,1].astype(int))])
data[0][1].append(['tender_final_total',np.sum(np.array(data[0][1])[:,1].astype(int))])
data[1][0].append(['bid_test_total',np.sum(np.array(data[1][0])[:,1].astype(int))])
data[1][1].append(['bid_final_total',np.sum(np.array(data[1][1])[:,1].astype(int))])


# 统计需要绘制的图表
plt_label = []
plt_data = []
plt_title = []
for i in range(len(data)):
    for j in range(len(data[i][0])):
        print(data[i][0][j][0])
        plt_title.append(f'{data[i][0][j][0]}**{data[i][0][j][1]}')

        # plt_title.append('_'.join([f"diff_{data[i][1][j][0]}",data[i][1][j][0]]))
        # print('_'.join([f"diff_{data[i][1][j][0]}",data[i][1][j][0]]))
        # print([data[i][1][j][1]/data[i][0][j][1],1-(data[i][1][j][1]/data[i][0][j][1])])
        plt_label.append([data[i][1][j][0],f"diff\n{data[i][0][j][1]-data[i][1][j][1]}"])
        plt_data.append([data[i][1][j][1]/data[i][0][j][1],1-(data[i][1][j][1]/data[i][0][j][1])])

# 绘制两个任务分表完成数量
for i in range(len(data)):
    label_ = []
    data_ = []
    for j in range(3):
        print(data[i][1][j][1])
        label_.append(data[i][1][j][0])
        data_.append(data[i][1][j][1]/data[i][0][3][1])
    label_.append(f'diff_total\n{data[i][0][3][1]-data[i][1][3][1]}')
    data_.append(1-sum(data_))
    plt_title.append(f"{tables[data[i][0][j][0]]['task']}**{data[i][0][j+1][1]}")
    plt_label.append(label_)
    plt_data.append(data_)

# 绘制统计图
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
fig = plt.figure(figsize=(30,20))
fig.patch.set_facecolor('white')
plt.rcParams.update({"font.size":18})#此处必须添加此句代码方可改变标题字体大小
for i in range(1,len(plt_data)):
    fig6 = plt.subplot(3,4,i)
    plt.title(plt_title[i-1])
    try:
        patches,l_text,p_text = \
        plt.pie(
            plt_data[i-1], 
            labels=plt_label[i-1], 
            autopct='%1.1f%%'
            )
        for i in p_text:
            i.set_fontsize(30)
    except:
        print(plt_data[i-1])
        print(plt_label[i-1])

fig6 = plt.subplot(3,4,11)
plt.title(plt_title[-1])
try:
    patches,l_text,p_text = \
    plt.pie(
        plt_data[-1], 
        labels=plt_label[-1], 
        autopct='%1.1f%%'
        )
    for i in p_text:
        i.set_fontsize(30)
except:
    print(plt_data[-1])
    print(plt_label[-1])
# 保存统计图  
plt.savefig('result.png', bbox_inches='tight')

