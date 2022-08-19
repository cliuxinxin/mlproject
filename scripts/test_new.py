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
                name = table + '#' + column + '#' + label + '#' + keyword
                train_name = name + '#train.json'
                # 如果train_name文件在assets目录下存在,则跳过执行
                if os.path.exists(ASSETS_PATH + train_name):
                    continue
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


files = os.listdir(ASSETS_PATH + 'bidclass/')

train = []
dev = []

for file in files:
    if file == '.DS_Store' or file == '.gitignore':
        continue
    paras = file.split('#')
    is_train = paras[-1] == 'train.json'
    df = pd.read_json(ASSETS_PATH + 'bidclass/' + file)
    if is_train:
        train.append(df[:10])
    else:
        dev.append(df[:5])

train = pd.concat(train)
dev = pd.concat(dev)

# 如果table,id,source,text,task,md5 都一样，则合并label column的值
train = train.groupby(['table','id','source','text','task','md5'])['label'].apply(lambda x:','.join(x)).reset_index()
dev = dev.groupby(['table','id','source','text','task','md5'])['label'].apply(lambda x:','.join(x)).reset_index()

train['len'] = train['text'].apply(lambda x:len(x))
dev['len'] = dev['text'].apply(lambda x:len(x))

# 去掉text为0的
train = train[train['len'] != 0]
dev = dev[dev['len'] != 0]
# 去掉text>50000的
train = train[train['len'] < 3000]
dev = dev[dev['len'] < 3000]
# 去掉text<50的
train = train[train['len'] > 50]
dev = dev[dev['len'] > 50]

b_save_df_datasets(train,'train_bid.json')
b_save_df_datasets(dev,'dev_bid.json')


train = b_read_dataset('train_bid.json')
dev = b_read_dataset('dev_bid.json')

for entry in train:
    labels = entry['label'].split(',')
    labels = set(labels)
    labels = list(labels)
    entry['label'] = labels

for entry in dev:
    labels = entry['label'].split(',')
    labels = set(labels)
    labels = list(labels)
    entry['label'] = labels

b_save_list_datasets(train,'train_bid_new.json')
b_save_list_datasets(dev,'dev_bid_new.json')

b_doccano_export_cat_project(37,'train.json')
b_doccano_export_cat_project(38,'dev.json')

df  = pd.DataFrame(train)
df['len'] = df['text'].apply(lambda x:len(x))
df.sort_values(by='len',ascending=False,inplace=True)

ls = list(range(11))

df = pd.DataFrame(ls)

df[:10]

