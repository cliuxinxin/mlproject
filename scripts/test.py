from tqdm import tqdm
from data_utils import *
from mysql_utils import *

df = pd.read_json(ASSETS_PATH + 'bid.json')

df = df['RECORDS'].apply(pd.Series)

df['text'] = df['detail_content'].apply(p_filter_tags)

df.text.values[0]
df.detail_content.values[0]
df.title.values[0]

nlp = b_load_best_model('bid')

doc = nlp(df.text.values[0])

for ent in doc.ents:
    print(ent.label_,ent.text)

sql ="SELECT fin.id,win.winning_bidder,win.table_name,win.amount,fin.publish_time,fin.source_website_name,fin.source_website_address FROM final_winning_bidder as win \
LEFT JOIN final_procurement_bid_result as fin on win.announcement_id = fin.id WHERE win.table_name = 'test_procurement_bid_result' AND DATE_FORMAT(fin.publish_time,'%Y-%m') = '2022-06' limit 1"

df = mysql_select_df(sql)

df = pd.read_csv(ASSETS_PATH + 'all.csv')

# 去掉amount是空的数据
df = df[~df.amount.isnull()]

# 如果 amount,tabel_name,announcement_id 有重复的数据
df = df.drop_duplicates(['amount','table_name','announcement_id'])

df_com = pd.read_csv(ASSETS_PATH + 'test_com_all.csv')


# 查看winner_bidder 的不同值的个数
df.winning_bidder.value_counts()

# 根据winnder_bidder统计行数和金额
df.groupby('winning_bidder').agg({'amount':'sum','table_name':'count'}).sort_values('table_name',ascending=False)


df[df.winning_bidder.isin(df_com.com.unique())].to_csv(ASSETS_PATH +  'test_com_all_2.csv')

df_2 = pd.read_csv(ASSETS_PATH + 'test_com_all_2.csv',index_col=0)

df_hands = pd.read_csv(ASSETS_PATH + 'hands.csv',index_col=0)

df_2['pk'] = df_2['table_name'] + '_' + df_2['announcement_id'] + '_' + df_2['winning_bidder']
df_hands['pk'] = df_hands['table_name'] + '_' + df_hands['announcement_id'] + '_' + df_hands['winning_bidder']

def find_amount(x):
    pk = x['pk']
    if pk in df_hands.pk.values:
        return df_hands.amount[df_hands.pk == pk].values[0]
    else:
        return df_2.amount[df_2.pk == pk].values[0]

def find_web_source_name(x):
    pk = x['pk']
    if pk in df_hands.pk.values:
        return df_hands.source_website_name[df_hands.pk == pk].values[0]
    else:
        return 0
import pandas as pd

# 根据pk，如果数据在df_hands中有，则将df_hands中的amount更新到df_2中的amount中
df_2['new_amount'] = df_2.apply(find_amount,axis=1)
df_2['amount_right'] = df_2.amount == df_2.new_amount

df_2['new_web_source_name'] = df_2.apply(find_web_source_name,axis=1)

df_2['commn'] = df_2['winning_bidder'] + '_' + df_2['amount'].astype(str)

df_2.to_csv(ASSETS_PATH + 'june.csv')

df_2[~df_2.source_website_name.isin(df_hands.source_website_name.unique())].source_website_name.value_counts()

df_june = pd.read_csv(ASSETS_PATH + 'june_new.csv',index_col=0)

# 找出new_amount不等于0的数据
df_june[df_june.new_amount != 0].drop_duplicates(['commn'],inplace=True)

df_june_0 = df_june[df_june.new_amount == 0]

df_june_not_0 = df_june[df_june.new_amount != 0]

df_june_not_0.drop_duplicates(['commn'],inplace=True)

df_june_new = pd.concat([df_june_0,df_june_not_0])

df_june_new.to_csv(ASSETS_PATH + 'june_new.csv')

df_june['web_url'] = df_june['source_website_address'] + "#:~:text=" + df_june['new_amount'].astype(str).str[:5]


df_june.to_csv(ASSETS_PATH + 'june_new_web.csv')


# 取new_amount前面5位数字
df_june['new_amount_5'] = df_june.new_amount.astype(str).str[:5]