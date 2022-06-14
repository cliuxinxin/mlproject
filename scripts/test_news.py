from data_utils import *
from mysql_utils import *
from data_clean_new import clean_manager
from google_utils import *

from gne import GeneralNewsExtractor

# 下载最新数据
b_doccano_export_project(14,'news.json')

news = b_read_dataset('news.json')
news = pd.DataFrame(news)

files = glob.glob(ASSETS_PATH + 'htmls/*')
extractor = GeneralNewsExtractor()
data = []

for file in files:
    entry = {}
    try:
        html = open(file, 'r').read()
        result = extractor.extract(html)
        title = result['title']
        entry['title'] = title
        content = result['content']
        entry['data'] = title + '\n' + content
        entry['md5'] = p_generate_md5(content)
        data.append(entry)
    except:
        continue

data = pd.DataFrame(data)

data = data[~data.md5.isin(news.md5)]

b_save_df_datasets(data, 'news_test.json')

b_doccano_upload('news_test.json',14)

news = news[:52]
news = news[52:]

# 重新随机排序
news = news.sample(frac=1)
len(news)

train = news[:int(len(news)*0.8)]
dev = news[int(len(news)*0.8):]

b_save_df_datasets(train, 'train.json')
b_save_df_datasets(dev, 'dev.json')
b_save_df_datasets(news, 'train_dev.json')

gdrive_upload_train_dev()

train = b_read_dataset('train.json')

train_dev_label = b_read_dataset('train_dev_label.json')

train_dev_label[0]

data = []

for entry in train_dev_label:
    md5 = entry['md5']
    text = entry['data']
    for start,end,label_ in entry['label']:
        new_entry = {}
        new_entry['md5'] = md5
        new_entry['start'] = start
        new_entry['label_'] = text[start:end]
        data.append(new_entry)

data = pd.DataFrame(data)

data.loc(data.md5)

