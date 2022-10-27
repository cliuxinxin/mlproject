from data_utils import * 
from mysql_utils import *
from redis_utils import pop_redis_data,redis_push,tag_key,sql_key,redis_
import time

class Helper():
    def __init__(self) -> None:
        self.tendercats = b_load_best_model('tendercats')
        self.bidcats = b_load_best_model('bidcats')
        self.contractcats = b_load_best_model('contractcats')
        self.tender_label = pd.DataFrame(b_read_dataset('tendercats_train_dev.json'))
        self.bid_label = pd.DataFrame(b_read_dataset('bidcats_train_dev.json'))
        self.contract_label = pd.DataFrame(b_read_dataset('contractcats_train_dev.json'))
    def get_model(self,task):
        if task == 'tendercats':
            return self.tendercats
        elif task == 'bidcats':
            return self.bidcats
        elif task == 'contractcats':
            return self.contractcats    
        else:
            # 报错
            raise Exception('没有指定任务')

    def get_label(self,task):
        if task == 'tendercats':
            return self.tender_label
        elif task == 'bidcats':
            return self.bid_label
        elif task == 'contractcats':
            return self.contract_label

def get_tag(doc):
    tag=sorted(doc.cats.items(),  key=lambda d: d[1],reverse=True)
    return tag[0][0]

def find_labels_by_md5(md5,label_data):
    """
    根据md5查找标签，填充到label里面
    """
    try:
        labels = label_data.loc[md5]['cats']
    except:
        labels = []
    return labels

helper = Helper()

while True:
    while len(redis_.keys(tag_key)) > 0:
        data = pop_redis_data(tag_key,200)
        # redis中还有多少数据
        print(f'There is {redis_.llen(tag_key)} data in redis')
        df = pd.DataFrame(data)
        df_group = df.groupby('table')
        for table,idxs in df_group.groups.items():
            df_table = df.loc[df_group.groups[table]]
            print(f"Process {table} {len(df_table)} records")
            sql = f"select id,detail_content from {table} where id in {tuple(df_table.id.values.tolist())}"
            task = df.loc[df_group.groups[table]].task.values[0]
            df1 = mysql_select_df(sql)
            df1 = df1[~df1['detail_content'].isnull()]
            df1['text'] = df1['detail_content'].apply(p_filter_tags)
            df1['md5'] = df1['text'].apply(p_generate_md5)
            df1['table'] = table
            text = df1.text.values.tolist()
            docs = helper.get_model(task).pipe(text)
            tags = []
            for doc in docs:
                tags.append(get_tag(doc))
            df1['classify_type'] = tags
            label_data = helper.get_label(task)
            df1.loc[df1.md5.isin(label_data.index),'classify_type'] = df1[df1.md5.isin(label_data.index)]['md5'].apply(lambda x:find_labels_by_md5(x,label_data))
            df1 = df1[['table','id','classify_type']]
            redis_push(df1,sql_key)
            print(f"Push {table} {len(df1)} records to redis")
    print("Sleep 5s")
    time.sleep(5)
    