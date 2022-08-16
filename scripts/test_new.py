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
    print(table)
    for column in columns:
        print(column)
        for label,keyword_list in keywords.items():
            print(label)
            for keyword in keyword_list:
                print(keyword)
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
                name = table + '#' + column + '#' + label + '#' + keyword
                if len_df == 1:
                    df.to_json(ASSETS_PATH + name + '#train.json')
                    df.to_json(ASSETS_PATH + name + '#dev.json')
                else:
                    df[:int(len_df*0.8)].to_json(ASSETS_PATH + name + '#train.json')
                    df[int(len_df*0.8):].to_json(ASSETS_PATH + name + '#dev.json')

