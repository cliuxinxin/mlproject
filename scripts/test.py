from data_utils import *
from mysql_utils import *
import time
import glob
from copy import deepcopy

# files = glob.glob(DATA_PATH + '*.json')

# tst_df = pd.read_json(files[0])

# ids = tst_df['id'].to_list()

# df = mysql_select_data_by_ids(ids,project_configs['tender']['table'])


sql = 'select * from tender_test'

new_df = mysql_select_df(sql)
df = deepcopy(new_df)   

nlp = b_load_best_model()
task = 'tender'
cols = df.columns.to_list()
std_labels = b_read_db_labels(task)
std_labels['col_idx'] = std_labels['col'].apply(lambda x: cols.index(x))


for i in range(len(df)):
  text = df.iloc[i][project_configs[task]['col']]
  id = df.iloc[i]['id']
  text = p_filter_tags(text)
  doc = nlp(text)
  for ent in doc.ents:
    col = std_labels[std_labels['label'] == ent.label_].iloc[0]['col_idx']
    df.iloc[i,col] = ent.text

comp = dc.Compare(df,new_df,join_columns=['id'])

print(comp.report())




