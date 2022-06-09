from data_utils import *
from mysql_utils import *
from data_clean_new import clean_manager

def find_labels_by_md5(md5,label_data):
    """
    根据md5查找标签，填充到label里面
    """
    try:
        labels = label_data.loc[md5]['labels']
    except:
        labels = []
    return labels

def get_labels(series):
    """
    整理标签
    """
    text = series['data']
    new_labels = []
    labels = series['labels']
    if len(labels) == 0:
        return new_labels
    for start,end,label_ in labels:
        new_label = {}
        new_label[label_] = text[start:end]
        new_labels.append(new_label)
    return new_labels

def clean_labels(labels,task):
    """
    根据任务和label名称清洗label
    """
    clean_labels = []
    for label in labels:
        clean_label = {}
        label_,text = list(label.items())[0]
        clean_text = clean_manager(task,label_,text) 
        clean_label[label_] = clean_text  
        clean_labels.append(clean_label)
    return clean_labels

def process_labels(labels,task):
    """
    根据任务做一些label之间的数据处理
    """
    if len(labels) == 0:
        return labels
    if task == 'bid':
        # 找到labels中的中标金额和中标金额单位
        unit = 1
        for label in labels:
            label_,text = list(label.items())[0]
            if label_ == '中标金额':
                amount = text
            if label_ == '中标金额单位':
                unit = text
        # 把中标金额和中标金额单位拼接起来
        new_labels = []
        for label in labels:
            new_label = {}
            label_,text = list(label.items())[0]
            new_label[label_] = text
            if label_ == '中标金额':
                new_label[label_] = float(amount) * float(unit)
            new_labels.append(new_label)
        return new_labels
    return labels

def preprocess_save(labels,task):
    """
    根据任务做多个labels的处理。
    """
    new_labels = {}
    if len(labels) == 0:
        return {}
    if task == 'bid':
        mult_labels = ['中标单位']
    for label in labels:
        label_,text = list(label.items())[0]
        if label_ in mult_labels and label_ in new_labels:
            new_labels[label_] = new_labels[label_] + '#' + text
        else:
            new_labels[label_] = text
    return new_labels
    

class Helper():
    def __init__(self) -> None:
        self.tender = b_load_best_model('tender')
        self.bid = b_load_best_model('bid')
        self.tender_label = pd.DataFrame(b_read_dataset('tender_train_dev.json'))
        self.bid_label = pd.DataFrame(b_read_dataset('bid_train_dev.json'))

    def get_model(self,task):
        if task == 'tender':
            return self.tender
        elif task == 'bid':
            return self.bid

    def get_label(self,task):
        if task == 'tender':
            return self.tender_label
        elif task == 'bid':
            bid_label = self.bid_label[['md5','label']]
            bid_label.columns = ['md5','labels']
            bid_label.set_index('md5',inplace=True)
            return bid_label

data_process = json.loads(open('data_process.json','r',encoding='utf-8').read())
helper = Helper()

files = glob.glob(DATA_PATH + '*.json')
file = files[0]

task = file.split('#')[0].split('/')[-1]
origin_table = file.split('#')[1]
target_table = data_process.get(origin_table).get('target')

label_data = helper.get_label(task)
nlp = helper.get_model(task)

df = pd.read_json(file)
df['data'] = df['detail_content'].fillna('')
df['data'] = df['data'].apply(p_filter_tags)
df['md5'] = df['data'].apply(p_generate_md5)
max_len = 10000
df['data'] = df['data'].str[:max_len]
# 计算标签
data = df['data'].to_list()
docs = nlp.pipe(data)
labels = []

for doc in docs:
    label = []
    for ent in doc.ents:
        label.append([ent.start_char,ent.end_char,ent.label_])
    labels.append(label)

df['labels'] = labels
# 替换正确标签
df[df.md5.isin(label_data.index)]['labels'] = df['md5'].apply(lambda x:find_labels_by_md5(x,label_data))
# 得到label内容
df['labels'] = df.apply(get_labels,axis=1)
# label根据任务做清洗
df['clean_labels'] = df['labels'].apply(lambda x:clean_labels(x,task))
# 根据任务，处理label之间的问题
df['clean_labels'] = df['clean_labels'].apply(lambda x:process_labels(x,task))
# 根据任务预处理数据，合并标签内容，有全部合并成一个或者是取最前面的数据
df['clean_labels'] = df['clean_labels'].apply(lambda x:preprocess_save(x,task))

for entry in df['md5']