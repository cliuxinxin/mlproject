

        elif mode == 'single':
            df = pd.read_json(file)
            df,std_labels,html_col = preprocess_df(df,task) 
            nlp = b_load_best_model(task)
            for idx in range(len(df)):
                process_df(idx,df,html_col,nlp,std_labels,task,label_data)

        elif mode == 'thread':
            df = pd.read_json(file)
            df,std_labels,html_col = preprocess_df(df,task) 
            nlp = b_load_best_model(task)
            q = queue.Queue()
            for idx in range(len(df)):
                q.put(idx)
            thread_num = thread_num
            threads = []
            for j in range(thread_num):
                t = threading.Thread(target=work, args=(q,df,html_col,nlp,std_labels,task,label_data))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()


        if mode == 'process':
            with BaseManager() as manager:
                corpus = manager.PoolCorpus(task,file,label_data)

                with Pool() as pool:
                    pool.map(corpus.add, (i for i in range(len(corpus.get()))))

                df = corpus.get()

    BaseManager.register('PoolCorpus', PoolCorpus)

class PoolCorpus(object):

    def __init__(self,task,file,label_data):
        df = pd.read_json(file)
        self.task = task
        df,std_labels,html_col = preprocess_df(df,task)
        self.html_col = html_col
        self.df = df
        self.std_labels = std_labels
        self.label_data = label_data
        self.nlp = b_load_best_model(task)

    def add(self, i):
        process_df(i,self.df,self.html_col,self.nlp,self.std_labels,self.task,self.label_data) 


    def get(self):
        return self.df


def work(q,df,html_col,nlp,std_labels,task):
        while True:
            if q.empty():
                return
            else:
                idx = q.get()
                process_df(idx,df,html_col,nlp,std_labels,task)

def b_doccano_download_train_dev_label_view_wrong():
    """
    下载最新的数据，并且用最好的模型预测，将对不上的数据上传的demo项目进行查看。
    可以根据错误类型+标签的方式查看对应标签的错误情况
    """
    b_doccano_train_dev_nlp_label()
    b_compare_human_machine_label()

# 读取新的文件进入到basic中
def b_add_new_file_to_db_basic(file):
    df = b_file_2_df(file)
    db = b_read_db_basic()
    df_db = pd.concat([db,df],axis=0)
    # 根据md5排重
    df_db = df_db.drop_duplicates(subset='md5',keep='first')
    b_save_db_basic(df_db)

# 随机查看数据
def b_check_random(data,num):
    test = random.sample(data,num)
    for entry in test:
        labels = entry['label']
        text = entry['text']
        label = random.sample(labels,1)[0]
        print(text[label[0]:label[1]],label[2])

# 读取最好test模型
def b_load_best_test():
    return spacy.load('../training/model-best-test')


# 保存所有数据
def b_save_db_all(df):
    d_save_pkl(df,DATABASE_PATH + 'all.pkl')


# 读取训练集的标签情况
def b_read_train_label_counts():
    train = b_read_dataset('train.json')
    label_counts = b_label_counts(train)
    return label_counts

# 读取训练集的labels并且保存
def b_save_labels():
    label_counts = b_read_train_label_counts()
    l = list(label_counts.keys())
    d_save_file(l, ASSETS_PATH + "labels.txt")

# 合并数据集
def b_combine_datasets(files:list) -> list:
    datas = []
    for file in files:
        data = d_read_json(ASSETS_PATH + file)
        datas.append(data)
    # 把列表的列表合并成一个列表
    return list(itertools.chain.from_iterable(datas))

# 读取lockfile
def b_read_lock_file() -> list:
    return d_read_file(path=LOCK_FILE_PATH)

# 保存lockfile
def b_save_lock_file(file):
    d_save_file(file,path=LOCK_FILE_PATH)

# 抽取数据
def p_extract_data(db_selected) -> pd.DataFrame:
    # 循环db_selected，抽取数据
    dfs = []
    for idx,item in db_selected.iterrows():
        file_name = item['file_name']
        ids = item['id']
        # 抽取数据
        df = pd.read_csv('../assets/' + file_name)
        df = df.loc[ids]
        df['id'] = df.index
        df['file_name'] = file_name
        dfs.append(df)
    # 合并dfs
    df = pd.concat(dfs)
    return df

# 获得文件和index
def p_get_data_index(db) -> pd.DataFrame:
    # 根据文件名获得ids，得到{"file_name":file_name,"id":[1,2,3,4,5]}
    db_selected = db.groupby('file_name').agg({'id':list})
    # 去掉index
    db_selected = db_selected.reset_index()
    return db_selected


# 随机选择100个样本
def p_random_select(db,num=100) -> pd.DataFrame:
    db = db.sample(n=num)
    return db



# 返回新添加的文件
def p_get_new_files(lock_files, files) -> list:
    new_files = []
    for file in files:
        if file not in lock_files:
            new_files.append(file)
    return new_files


def test():
    # 抽取已标注数据的标签情况
    b_doccano_dataset_label_view('train.json',['招标项目编号'],1)
    # 从未标注数据中选取数据
    db = b_extrct_data_from_db_basic('tender')
    df_db = pd.DataFrame(db)
    # 查看未标注数据的关键词情况
    b_doccano_cat_data(df_db,100,['招标编号','招标项目编号'])
    # 保存未标注数据
    b_save_df_datasets(df_db,'test2.json')
    # 模型标注数据
    b_label_dataset('test2.json')
    # 合并标注数据标签和原始数据
    
    # 将json训练数据转换为bio标签
    bio_labels = b_generate_biolabels_from('labels.txt')
    # 将json训练数据转换为bio训练数据 train_trf.json文件
    b_trans_dataset_bio(bio_labels,'train.json')
    b_trans_dataset_bio(bio_labels,'dev.json')
    # 划分数据集为更小的集合
    b_bio_split_dataset_by_max('train_trf.json',510)
    b_bio_split_dataset_by_max('dev_trf.json',510)



# 计算分割点
def cal_boder_end(block,length=4800):
    borders_end = []
    # 计算出边界值
    for i in range(block):
        borders_end.append((i+1) * length)
    return borders_end

# 分割标签
def cal_border_and_label_belong(labels,borders_end):
    label_loc = []
    for label in labels:
        start = label[0]
        end = label[1]
        for idx,border in enumerate(borders_end):
            if start < border and end < border:
                label_loc.append(idx)
                break
            if start < border and end > border:
                pad = end - border + 20
                label_loc.append(idx)
                for idxc,border in enumerate(borders_end):
                    if idxc >= idx:
                        borders_end[idxc] += pad
                break
    return label_loc

# 拆分数据集
def generate_new_datasets(new_data, text, labels, borders_end, borders_start, label_loc,id):
    idx = 0
    for b_start,b_end in zip(borders_start,borders_end):
        entry = {}
        entry['text'] = text[b_start:b_end]
        new_labels = []
        for idxl,loc in enumerate(label_loc):
            if loc == idx:
                label = labels[idxl].copy()
                label[0] -= b_start
                label[1] -= b_start
                new_labels.append(label)
        entry['label'] = new_labels
        entry['id'] = id
        idx += 1
        if len(new_labels) != 0:
            new_data.append(entry)

# 分割数据集
def cut_datasets_size_pipe():
    data = read_datasets('train.json')

    new_data = []
    for entry in data:
        id = entry['id']
        text = entry['text']
        labels = entry['label']
        if len(text) < 4800:
            new_data.append(entry)
        else:
            blcok = math.ceil(len(text) / 4800)
            borders_end = cal_boder_end(blcok)
            borders_start = [0] + [i + 1 for i in borders_end[:-1]]
            label_loc = cal_border_and_label_belong(labels,borders_end)
            generate_new_datasets(new_data, text, labels, borders_end, borders_start, label_loc,id)

    df = pd.DataFrame(new_data)
    save_datasets(df,file_name='new_train.json',is_label=True)

    # 合并训练集和测试集
def combin_datasets(train_file="train.json",dev_file="dev.json"):
    # 读取train.json
    train = read_datasets(train_file)
    # 读取dev.json
    dev = read_datasets(dev_file)
    # 合并
    train.extend(dev)
    return train


def train_data_evaluate():
    train = combin_datasets(train_file="train.json",dev_file="dev.json")

    nlp = spacy.load('../training/model-best')

    text_data = [entry['text'] for entry in train]

    docs = nlp.pipe(text_data)

    for idx,doc in enumerate(docs):
        entry = train[idx]
        label_p = [[ent.text,ent.start,ent.end,ent.label_] for ent in doc.ents]
        label = entry['label']
        new_labels = []
        for item in label:
            new_label = []
            start = item[0]
            end = item[1]
            label_ = item[2]
            text = entry['text'][start:end]
            new_label.append(text)
            new_label.append(start)
            new_label.append(end)
            new_label.append(label_)
            new_labels.append(new_label)
        entry['label_p'] = label_p
        entry['new_label'] = new_labels
    return train


def label_right_counts(train):
    # 统计train当中label_p标注正确的个数
    label_p_count = {}
    for entry in train:
        label_p = entry['label_p']
        new_label = entry['new_label']
        for item in label_p:
            label_ = item[3]
            text = item[0]
            for new_item in new_label:
                if new_item[0] == text and new_item[3] == label_:
                    if label_ in label_p_count:
                        label_p_count[label_] += 1
                    else:
                        label_p_count[label_] = 1
    return label_p_count

def label_acc(label_count, label_p_count):
    # 通过label_count 和 label_p_count 计算准确率
    label_accuracy = {}
    for label_,count in label_count.items():
        if label_ in label_p_count:
            label_accuracy[label_] = label_p_count[label_] / count
        else:
            label_accuracy[label_] = 0
    return label_accuracy

df = b_file_2_df('Untitled.csv',text_col="details")

    # 保存数据库
    b_save_database_all(df)

    # 选择所有的数据
    df = p_get_data_index(df)

    # 提取数据
    df = p_extract_data(df)

    # 预处理
    p_html_text(df,'details')

    # 提取details中包含 “招标编号”的数据
    df_t = df[df['details'].str.contains('招标编号')]

    # 随机调整顺序
    df_t = df_t.sample(frac=1)

    # 抽取500个数据
    df_t = df_t.head(500)

    # 新增列dataset，前面50个为 tender_dev , 后面 450个为 tender_train
    df_t['dataset'] = 'tender_train'
    df_t.iloc[0:50,df_t.columns.get_loc('dataset')] = 'tender_dev'

    # 统计 dataset 个数
    df_t['dataset'].value_counts()

    # 将details改名为text
    df_t = df_t.rename(columns={'details':'text'})

    # 保存数据集
    tender_train  = df_t[df_t['dataset'] == 'tender_train']
    tender_dev = df_t[df_t['dataset'] == 'tender_dev']

    
    # 保存数据集
    with open('../assets/tender_train.json','w',encoding='utf-8') as f:
        for entry in tender_train.to_dict('records'):
            json.dump(entry,f,ensure_ascii=False)
            f.write('\n')

    # 保存测试集
    with open('../assets/tender_dev.json','w',encoding='utf-8') as f:
        for entry in tender_dev.to_dict('records'):
            json.dump(entry,f,ensure_ascii=False)
            f.write('\n')

    # 去掉text
    df_t = df_t.drop(columns=['text'])

    # 保存数据库
    b_save_database_datasets(df_t)


    def cut_sent(txt):
    txt = re.sub('([。])',r"\1\n",txt) # 单字符断句符，加入中英文分号
    txt = txt.rstrip()       # 段尾如果有多余的\n就去掉它
    nlist = txt.splitlines()
    nlist = [x for x in nlist if x.strip()!='']  # 过滤掉空行
    return nlist

# 读取训练文件
data =b_read_dataset('train.json')


ret = []
for entry in data:
    txt = entry['text']
    # 分句
    sents = cut_sent(txt)
    total = len(sents)
    # 计算每句的长度
    sents_segment = list(map(len, sents))
    # 累加得到每句的起止位置
    pos = np.cumsum(sents_segment)
    sents_pos = list( zip( [0] + pos.tolist(), pos))


    # 读取标签的位置
    labels = entry['label']
    labels_pos = [x[:2] for x in labels]
    # 按位置排序一下
    labels_pos = sorted(labels_pos, key=lambda x:x[0])


    labels_count = len(labels_pos)


    # 遍历两个位置，去掉没有重叠的部分
    i,j=0,0
    spos = sents_pos[i]
    lpos = labels_pos[j]
    result = []
    retdat = {}
    while 1:
        # 判断spos 和 lpos 的交叉情况
        if lpos[1]<=spos[0]: # 标签在句子左，继续移动标签
            j+=1
        if spos[1]<=lpos[0]: # 句子在标签左，继续移动句子
            i+=1

        if spos[0]<=lpos[1]<=spos[1] :
            # 加字典的方式返回
            if i in retdat.keys():
                retdat[i].append((lpos,j))
            else:
                retdat[i] = [(lpos,j)]
            j+=1
            
        if lpos[0]<=spos[1]<=lpos[1] :
            #result.append([spos,lpos,i,j])
            if i in retdat.keys():
                retdat[i].append((lpos,j))
            else:
                retdat[i] = [(lpos,j)]
            i+=1
            #j+=1
        if i>=total: break
        if j>=labels_count: break
        spos = sents_pos[i]
        lpos = labels_pos[j]
    

    for k,v in retdat.items():
        sb,se = sents_pos[k] # 句子位置
        sent_txt = txt[sb:se]
        sub_labels = []
        for (lb,le), j in v:
            lbltxt = txt[lb:le]
            lbl_label = labels[j][2]
            sub_labels.append([lb-sb,le-sb,lbl_label])

        ret.append({'text': sent_txt, 'label':sub_labels})

b_save_list_datasets(ret,'train.json')



