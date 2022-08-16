from mysql_utils import *
from data_utils import *

tables = ['test_other_tender_bid_result','test_procurement_bid_result','test_tender_bid_result']

project_type = json.load(open(ASSETS_PATH + 'project_type.json'))

keywords = {}
for key,value in project_type.items():
    lable = key.split('_')[0]
    keyword = key.split('_')[1]
    if lable not in keywords:
        keywords[lable] = [keyword]
    else:
        keywords[lable].append(keyword)

keywords['EPC'] = ['EPC']

table = tables[0]

columns = ['title','detail_content']

train = []
dev = []

for table in tables:
    for column in columns:
        for label,keyword_list in keywords.items():
            for keyword in keyword_list:
                sql = f"select id,source_website_address,detail_content from {table} where {column} regexp('{keyword}') limit 100"
                try:
                    df = mysql_select_df(sql)
                except:
                    continue
                len_df = len(df)
                if len_df == 0:
                    continue
                df['table'] = table
                df['label'] = label
                df = df[['table','id','source_website_address','detail_content','label']]
                df = p_process_df(df,'bidcats')
                if len_df == 1:
                    train.append(df)
                    dev.append(df)
                else:
                    train.append(df[:int(len_df*0.8)])
                    dev.append(df[int(len_df*0.8):])

df_train = pd.concat(train)
df_dev = pd.concat(dev)

b_save_df_datasets(df_train,'train_cats.json')
b_save_df_datasets(df_dev,'dev_cats.json')
