from data_utils import * 
from mysql_utils import *
from redis_utils import pop_redis_data,redis_push,tag_key,sql_key
import time

class Helper():
    def __init__(self) -> None:
        self.tendercats = b_load_best_model('tendercats')
        self.bidcats = b_load_best_model('bidcats')
        # self.tender_label = pd.DataFrame(b_read_dataset('tender_train_dev.json'))
        # self.bid_label = pd.DataFrame(b_read_dataset('bid_train_dev.json'))

    def get_model(self,task):
        if task == 'tendercats':
            return self.tendercats
        elif task == 'bidcats':
            return self.bidcats
        else:
            # 报错
            raise Exception('没有指定任务')

def get_tag(doc,threhold):
    tag = []
    for label,prob in doc.cats.items():
        if prob > threhold:
            tag.append(label)
    if len(tag) == 0:
        tag = ['其他']
    return ','.join(tag)

helper = Helper()
threhold = 0.7

while True:
    data = pop_redis_data(tag_key,200)
    df = pd.DataFrame(data)
    df_group = df.groupby('table')
    for table,idxs in df_group.groups.items():
        sql = f"select id,detail_content from {table} where id in {tuple(df.loc[df_group.groups[table]].id.values.tolist())}"
        task = df.loc[df_group.groups[table]].task.values[0]
        df1 = mysql_select_df(sql)
        df1['text'] = df1['detail_content'].apply(p_filter_tags)
        df1['table'] = table
        text = df1.text.values.tolist()
        docs = helper.get_model(task).pipe(text)
        tags = []
        for doc in docs:
            tags.append(get_tag(doc,threhold))
        df1['classify_type'] = tags
        df1 = df1[['table','id','classify_type']]
        redis_push(df1,sql_key)
    time.sleep(5)
    


    
    