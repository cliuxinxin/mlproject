from data_utils import *
from mysql_utils import *
import time
import glob

task = 'tender'
table = project_configs[task]['table']
html_col = project_configs[task]['col']

sql = "select * from %s  order by create_time desc limit %s" % (table,100)

# 查询 create_time 在 60 分钟以内的数据
sql = "select * from %s where create_time > now() - interval 180 minute order by create_time desc limit 100" % (table)

df = mysql_select_df(sql)

# 每10条数据保存成一个单独的文件
for idx,i in enumerate(range(0,len(df),10)):
  # 文件名为处理数据的时间戳,精确到厘秒
  file_name = str(int(time.time()*100000))
  df[i:(i+10)].to_json(DATA_PATH + file_name+'.json')


# 取第一行数据
tst_df.iloc[0]

db = b_read_db_labels()

class PoolCorpus(object):

    def __init__(self,task,file):
        df = pd.read_json(file)
        cols = df.columns.to_list()
        std_labels = b_read_db_labels(task)
        std_labels['col_idx'] = std_labels['col'].apply(lambda x: cols.index(x))
        self.html_col = project_configs[task]['col']
        self.task = task
        self.df = df
        self.std_labels = std_labels
        self.nlp = b_load_best_model(task)

    def add(self, i):
        text = self.df.iloc[i][self.html_col]
        id = self.df.iloc[i]['id']
        text = p_filter_tags(text)
        doc = self.nlp(text)
        for ent in doc.ents:
          col = self.std_labels[self.std_labels['label'] == ent.label_].iloc[0]['col_idx']
          self.df.iloc[i,col] = ent.text
        mysql_delete_data_by_id(id,self.task)

    def get(self):
        return self.df

BaseManager.register('PoolCorpus', PoolCorpus)

with BaseManager() as manager:
    corpus = manager.PoolCorpus('tender',files[0])

    with Pool() as pool:
        pool.map(corpus.add, (i for i in range(len(corpus.get()))))

    new_data = corpus.get()

# 将文件移动到已处理目录
files = glob.glob(DATA_PATH + '*.json')
file_name = files[0].split('/')[-1]
os.rename(files[0],DATA_PATH + 'processed/' + file_name)

mysql_insert_data(new_data,'tender')


files = glob.glob(DATA_PATH + '*.json')

tst_df = pd.read_json(files[0])

ids = tst_df['id'].to_list()

df = mysql_select_data_by_ids(ids,project_configs['tender']['table'])

sql = 'select * from tender_test'

new_df = mysql_select_df(sql)

import datacompy as dc

comp = dc.Compare(tst_df,new_df,join_columns=['id'])

print(compare.report())


from copy import deepcopy

nlp = b_load_best_model()

cols = df.columns.to_list()
std_labels = b_read_db_labels(task)
std_labels['col_idx'] = std_labels['col'].apply(lambda x: cols.index(x))
task = 'tender'
df = deepcopy(new_df)
for i in range(len(df)):
  text = df.iloc[i][project_configs[task]['col']]
  id = df.iloc[i]['id']
  text = p_filter_tags(text)
  doc = nlp(text)
  for ent in doc.ents:
    col = std_labels[std_labels['label'] == ent.label_].iloc[0]['col_idx']
    df.iloc[i,col] = ent.text

