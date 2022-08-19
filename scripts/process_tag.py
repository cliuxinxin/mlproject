from data_utils import * 
from mysql_utils import *
from redis_utils import pop_redis_data,redis_push,tag_key,sql_key

task = 'bidcats'
threhold = 0.7

def get_tag(doc,threhold):
    tag = []
    for label,prob in doc.cats.items():
        if prob > threhold:
            tag.append(label)
    if len(tag) == 0:
        tag = ['其他']
    return ','.join(tag)

nlp = b_load_best_model(task)

data = pop_redis_data(tag_key,10)

ids = [entry['id'] for entry in data]

sql = f"select id,source_website_address,detail_content from {table} where id in {tuple(ids)}"
df = mysql_select_df(sql)
df = p_process_df(df,task)
df['table'] = table
text = df.text.values.tolist()
docs = nlp.pipe(text)
tags = []
for doc in docs:
    tags.append(get_tag(doc,threhold))
df['tags'] = tags
df = df[['table','id','tags']]
redis_push(df,sql_key)
    
    