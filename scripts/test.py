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


