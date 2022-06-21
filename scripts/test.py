from data_utils import *
from mysql_utils import *
from data_clean_new import clean_manager
import cv2


import pandas as pd

files = glob.glob(ASSETS_PATH + 'dorab/*.jpg')

file = files[3]

img = cv2.imread(file)

sp = img.shape

file

sql1 = 'select a.announcement_id,a.amount,a.winning_bidder,b.source_website_address,b.labels from final_winning_bidder a left join final_tender_bid_result b on a.announcement_id = b.id and b.source_website_address is not null order by a.amount desc limit 1000'
sql2 = 'select a.announcement_id,a.amount,a.winning_bidder,b.source_website_address,b.labels from final_winning_bidder a left join final_tender_bid_result b on a.announcement_id = b.id and b.source_website_address is not null order by a.amount limit 1000'

df1 = pd.read_sql(sql1, con=conn)
df2 = pd.read_sql(sql2, con=conn)

df = pd.concat([df1,df2])

df = df[df.source_website_address.notnull()]

df['source_website_address'] = df['source_website_address']  + "#:~:text=" + df['winning_bidder']

df = df.sort_values(by='announcement_id', ascending=False)
df = df.drop_duplicates(subset=['announcement_id'], keep='first')

df['human_check'] = ''

df.to_csv(ASSETS_PATH + 'BIG_AMOUNT.csv', index=False)