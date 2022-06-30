from data_utils import *
from mysql_utils import *
from data_clean_new import clean_manager
import cv2
import numpy as np


sql1 = 'select a.announcement_id,a.amount,a.winning_bidder,b.source_website_address,b.labels,"final_tender_bid_result" as tabel_name from final_winning_bidder a left join final_tender_bid_result b on a.announcement_id = b.id and b.source_website_address is not null order by a.amount desc limit 2000'
sql2 = 'select a.announcement_id,a.amount,a.winning_bidder,b.source_website_address,b.labels,"final_tender_bid_result" as tabel_name from final_winning_bidder a left join final_tender_bid_result b on a.announcement_id = b.id and b.source_website_address is not null order by a.amount limit 2000'
sql3 = 'select a.announcement_id,a.amount,a.winning_bidder,b.source_website_address,b.labels,"final_other_tender_bid_result" as tabel_name from final_winning_bidder a left join final_other_tender_bid_result b on a.announcement_id = b.id and b.source_website_address is not null order by a.amount limit 2000'
sql4 = 'select a.announcement_id,a.amount,a.winning_bidder,b.source_website_address,b.labels,"final_other_tender_bid_result" as tabel_name from final_winning_bidder a left join final_other_tender_bid_result b on a.announcement_id = b.id and b.source_website_address is not null order by a.amount limit 2000'

sql1 = 'select a.announcement_id,a.amount,a.winning_bidder,b.source_website_address,b.labels,"final_procurement_bid_result" as tabel_name from final_winning_bidder a left join final_procurement_bid_result b on a.announcement_id = b.id and b.source_website_address is not null order by a.amount desc limit 1000'




df1 = pd.read_sql(sql1, con=conn)
df2 = pd.read_sql(sql2, con=conn)
df3 = pd.read_sql(sql3, con=conn)
df4 = pd.read_sql(sql4, con=conn)

df = pd.concat([df1,df2,df3,df4])

df = df1

df = df[df.source_website_address.notnull()]

df['source_website_address'] = df['source_website_address']  + "#:~:text=" + df['winning_bidder']

df = df.sort_values(by='announcement_id', ascending=False)
df = df.drop_duplicates(subset=['announcement_id'], keep='first')

df['human_check'] = ''

df_db = pd.read_csv(ASSETS_PATH + '20220627.csv')

df.to_csv(ASSETS_PATH + '20220627.csv', index=False)

df[~df.announcement_id.isin(df_db.announcement_id)].to_csv(ASSETS_PATH + '20220622.csv', index=False)

dora_path = ASSETS_PATH + 'dora/'
dorab_path = ASSETS_PATH + 'dorab/'


dorab_paths = glob.glob(dorab_path + '*')

test_path = dorab_paths[11]

# test_path '../assets/dorab/001 - 第1卷'
# 取得vol = 001
vol = test_path.split('/')[-1].split(' - ')[0][1:]

dora_files = glob.glob(dora_path + '*.jpg')
dorab_files = glob.glob(test_path + '/*.jpg')

# 随机取出dorab_files中的一个文件
#  '../assets/dorab/001 - 第1卷/71.jpg',
# 找到对应 dora_files中的文件 '../assets/dora/01_071.jpg',
page = dorab_files[0].split('/')[-1].split('.')[0]
# 扩充为3位
page = '0' + page if len(page) == 2 else page
dora_file = '../assets/dora/' + vol + '_' + page + '.jpg'
dorab_file = dorab_files[0]

# 并排显示两个图片
img1 = cv2.imread(dora_file)
img2 = cv2.imread(dorab_file)

# 调整同样高
img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

# 合并两个图片
img = np.concatenate([img1, img2], axis=1)

# 显示图片
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 按照数字排序
dorab_files = sorted(dorab_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

# 修改名字，从2开始
for i, file in enumerate(dorab_files):
    new_name = str(i + 1).zfill(3) + '.jpg'
    os.rename(file, test_path + '/' + new_name)

def label_data(nlp,data):
    doc = nlp(data)
    return [[ent.start_char,ent.end_char,ent.label_] for ent in doc.ents]

df = pd.read_excel(ASSETS_PATH + '20220628.xlsx')

# test_tender_bid_result test_bid
# test_other_tender_bid_result test_other

ids = df['announcement_id'].tolist()
table = 'test_other_tender_bid_result' 
col = 'detail_content'
source = 'source_website_address'
task = 'bid'
file_name = task + '#' + table + '#' + str(int(time.time()*100000))


# sql = "select * from {} where id in {}".format(table,tuple(ids))
sql = "select * from final_other_tender_bid_result where id='zzlh-328322'"

df = pd.read_sql(sql, con=conn)

df.to_json(DATA_PATH + file_name + '.json')
df = df[[col,source]]
df = p_process_df(df,task)
nlp = b_load_best_model(task)
df['label'] = df['data'].apply(lambda x:label_data(nlp,x))
b_save_df_datasets(df,'one_data.json')
b_doccano_upload_by_task('one_data.json',task,'train')
print("Get %s data from mysql and save to json" % len(df))

data = df['data'].to_list()
docs = nlp.pipe(data)
labels = []

for doc in docs:
    label = []
    for ent in doc.ents:
        label.append([ent.start_char,ent.end_char,ent.label_])
    labels.append(label)

df['label'] = labels

# final_winning_bidder 有 announcement_id 和 winning_bidder
# 查出 announcement_id 和 winning_bidder 重复的数据
sql = "select table_name,announcement_id,winning_bidder,count(1) from final_winning_bidder group by table_name,announcement_id,winning_bidder having count(1) > 1"

df = pd.read_sql(sql, con=conn)

df.drop_duplicates(subset=['table_name','announcement_id','winning_bidder'], keep='first', inplace=True)

len(df)

df1 = df[df.table_name=='test_other_tender_bid_result']
df2 = df[df.table_name=='test_tender_bid_result']

df = df2

ids = df['announcement_id'].tolist()
table = 'test_tender_bid_result' 
col = 'detail_content'
source = 'source_website_address'
task = 'bid'
file_name = task + '#' + table + '#' + str(int(time.time()*100000))

sql = "select * from {} where id in {}".format(table,tuple(ids))

df = pd.read_sql(sql, con=conn)

df.to_json(DATA_PATH + file_name + '.json')



import pandas as pd

files = glob.glob(DATA_PATH + '*.json')

dfs = []

for file in files:
    df = pd.read_json(file)
    dfs.append(df)

df = pd.concat(dfs)

df['labels']
df['data'] = df['detail_content'].fillna('')
df['data'] = df['data'].apply(p_filter_tags)
df['md5'] = df['data'].apply(p_generate_md5)

bid_label = pd.DataFrame(b_read_dataset('bid_train_dev.json'))
bid_label = bid_label[['md5','label']]
bid_label.columns = ['md5','labels']
bid_label.set_index('md5',inplace=True)
label_data = bid_label

def find_labels_by_md5(md5,label_data):
    """
    根据md5查找标签，填充到label里面
    """
    try:
        labels = label_data.loc[md5]['labels']
    except:
        labels = []
    return labels

df[df.md5.isin(label_data.index)]['labels'] = df[df.md5.isin(label_data.index)]['md5'].apply(lambda x:find_labels_by_md5(x,label_data))

df['labels']

df[df.md5.isin(label_data.index)]['md5'].apply(lambda x:find_labels_by_md5(x,label_data))

df.loc[df.md]
