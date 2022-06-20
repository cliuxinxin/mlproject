from data_utils import *
from google_utils import *

from gne import GeneralNewsExtractor
extractor = GeneralNewsExtractor()
nlp = b_load_best_model('news')

def get_html_files(path):
    """
    查找目录下面htm和html文件
    """
    html_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.html') or file.endswith('.htm'):
                if file.startswith('.'):
                    continue
                html_files.append(os.path.join(root, file))
    return html_files

def move_htmls(files):
    """
    移动到ASSETS_PATH/htmls目录下
    """
    for file in files:
        # 如果目录有重名的文件，则修改文件名
        if os.path.exists(os.path.join(ASSETS_PATH, 'htmls', file.split('/')[-1])):
            os.rename(file, os.path.join(ASSETS_PATH, 'htmls', file.split('/')[-1] + '_' + str(random.randint(0, 100))))
        else:
            os.rename(file, os.path.join(ASSETS_PATH, 'htmls', file.split('/')[-1]))

def label_data(data):
    """
    标注数据
    """
    doc = nlp(data)
    label = [[ent.start_char, ent.end_char, ent.label_] for ent in doc.ents]
    return label

# 下载目录
path = '/Users/liuxinxin/Downloads/'
# 查找目录下面htm和html文件
files = get_html_files(path)
# 移动到ASSETS_PATH/htmls目录下
move_htmls(files)

# 下载最新数据
b_doccano_export_project(14,'news.json')

news = b_read_dataset('news.json')
news = pd.DataFrame(news)

files = glob.glob(ASSETS_PATH + 'htmls/*')

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

data['label'] = data.data.apply(label_data)

b_save_df_datasets(data, 'news_test.json')

b_doccano_upload('news_test.json',14)



# 重新随机排序
news = news.sample(frac=1)

train = news[:int(len(news)*0.8)]
dev = news[int(len(news)*0.8):]

b_save_df_datasets(train, 'train.json')
b_save_df_datasets(dev, 'dev.json')
b_save_df_datasets(news, 'train_dev.json')

b_remove_invalid_label('train.json')
b_remove_invalid_label('dev.json')

gdrive_upload_train_dev()
