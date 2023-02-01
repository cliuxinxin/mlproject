from tqdm import tqdm
from data_utils import *
from mysql_utils import *
import glob
from data_clean_new import clean_manager,clean_money
from decimal import Decimal
from scrapy import Selector

def all_labels_is_empty(labels):
    """
    检查所有labels是否为空
    """
    for label in labels:
        if len(label) > 0:
            return False
    return True

def d_date_clean(value):
    """
    清洗日期
    """
    # 如果value是datetime.datetime类型，则直接返回
    if isinstance(value,datetime):
        if type(value) != pd._libs.tslibs.nattype.NaTType:
            return datetime.fromtimestamp(value.timestamp())

def datetime_process(df,task):
    """
    处理日期
    """
    if task == 'bid':
        datetime_columns = ['publish_time','publish_stime','publish_etime']
    elif task == 'tender':
        datetime_columns = ['publish_time','tender_document_stime','tender_document_etime','quote_stime','quote_etime','tender_stime','tender_etime','publish_stime','publish_etime','bid_opening_time']
    elif task=="contract":
        datetime_columns=['publish_time','publish_stime','publish_etime','start_time','end_time']
    else:
        datetime_columns = []
    for colum in datetime_columns:
        # 转换为datetime格式
        try:
            df[colum] = pd.to_datetime(df[colum])
            df[colum] = df[colum].apply(lambda x: d_date_clean(x))
        except:
            pass
        # 转换为时间戳

    return df

def move_file(file):
    """
    移动已处理文件
    """
    file_name = file.split('/')[-1]
    os.rename(file,DATA_PATH + 'processed/' + file_name)

def delete_win_by_df(target_table, df, origin_table):
    """
    删除数据
    """
    ids = df['announcement_id'].to_list()
    if len(ids) == 1:
        id = ids[0]
        mysql_delete_win_by_id(id,origin_table,target_table)
    else:
        mysql_delete_win_by_ids(ids,origin_table,target_table)

def delete_mysql_by_df(target_table, df):
    """
    删除数据
    """
    ids = df['id'].to_list()
    if len(ids) == 1:
        id = ids[0]
        mysql_delete_data_by_id(id,target_table)
    else:
        mysql_delete_data_by_ids(ids,target_table)

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
    text = series['text']
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
    根据任务做一些label之间的数据处理，将金额和金额单位转为数字
    """
    if len(labels) == 0:
        return labels
    # 找到labels中的金额和金额单位
    unit = 1
    for label in labels:
        label_,text = list(label.items())[0]
        if label_ in ['中标金额','预算']:
            amount = text
        if label_ in ['中标金额单位','预算单位']:
            unit = text
    # 把中标金额和中标金额单位拼接起来
    new_labels = []
    for label in labels:
        new_label = {}
        label_,text = list(label.items())[0]
        new_label[label_] = text
        if label_ in ['中标金额','预算']:
            new_label[label_] = Decimal(amount) * Decimal(str(unit))
        new_labels.append(new_label)
    return new_labels

def preprocess_save(labels,task):
    """
    根据任务做多个labels的处理。
    """
    new_labels = {}
    if len(labels) == 0:
        return {}
    mult_labels = []  
    if task == 'bid':
        mult_labels = ['中标单位']
    for label in labels:
        label_,text = list(label.items())[0]
        if label_ in mult_labels and label_ in new_labels:
            new_labels[label_] = new_labels[label_] + '#' + text
        else:
            new_labels[label_] = text
    return new_labels

def fill_data(x,label_df,col):
    """
    填充数据
    """
    try:
        result = label_df[label_df['md5']==x][col].values[0]
    except:
        result = ''
    return result

def process_save(data_process, task, origin_table, df):
    """
    处理数据
    """
    data = []
    sub_data = []

    for md5,labels,id,publish_time,money in df[['md5','clean_labels','id','publish_time','money']].values:
        if len(labels) == 0:
            continue
        df_labels = pd.DataFrame([labels])
        colunms = list(df_labels.columns)
        new_columns = []
        for column in colunms:
            col = data_process.get(task).get(column)
            if col:
                new_columns.append(col)
            else:
                new_columns.append(column)
        df_labels.columns = new_columns
        df_labels['md5'] = md5
        if task == 'bid':
            if money != 0:
                df_labels['amount'] =  money
            if 'winning_bidder' in df_labels.columns:
                df_labels['winning_bidder'] = df_labels['winning_bidder'].apply(lambda x: x.split('#'))
                split_df_labels = df_labels.explode('winning_bidder')
                # 根据winner_bidder排重
                split_df_labels = split_df_labels.drop_duplicates(subset=['winning_bidder'])
                split_df_labels[['announcement_id','table_name','publish_time']] = [id, origin_table, publish_time]
                if 'amount' in split_df_labels.columns and len(split_df_labels) > 1:
                # 保留第一行amount,其他行清空
                    split_df_labels.reset_index(inplace=True)
                    split_df_labels.loc[1:,'amount'] = 0
                sub_data.append(split_df_labels)
            df_labels[['winning_bidder']] = ''
            # 如果有中标金额单位，则删除
            if '中标金额单位' in df_labels.columns:
                df_labels.drop('中标金额单位',axis=1,inplace=True)
        if '预算单位' in df_labels.columns:
            df_labels.drop('预算单位',axis=1,inplace=True)
        data.append(df_labels)
        
    label_df = pd.concat(data)
    if task == 'bid' and len(sub_data) > 0:
        sub_label_df = pd.concat(sub_data)
        if 'amount' not in sub_label_df.columns:
            sub_label_df['amount'] = 0
        sub_label_df = sub_label_df[['winning_bidder','amount','table_name','announcement_id','publish_time']]
    else:
        sub_label_df = pd.DataFrame()
    return label_df,sub_label_df

class Helper():
    def __init__(self) -> None:
        self.tender = b_load_best_model('tender')
        self.bid = b_load_best_model('bid')
        self.contract = b_load_best_model('contract') 
        self.tender_label = pd.DataFrame(b_read_dataset('tender_train_dev.json'))
        self.bid_label = pd.DataFrame(b_read_dataset('bid_train_dev.json'))
        self.contract_label = pd.DataFrame(b_read_dataset('contract_train_dev.json'))

    def get_model(self,task):
        if task == 'tender':
            return self.tender
        elif task == 'bid':
            return self.bid
        elif task == 'contract':
            return self.contract
        else:
            # 报错
            raise Exception('没有指定任务')

    def get_label(self,task):
        if task == 'tender':
            tender_label = self.tender_label[['md5','label']]
            tender_label.columns = ['md5','labels']
            tender_label.set_index('md5',inplace=True)
            return tender_label
        elif task == 'bid':
            bid_label = self.bid_label[['md5','label']]
            bid_label.columns = ['md5','labels']
            bid_label.set_index('md5',inplace=True)
            return bid_label
        elif task == 'contract':
            contract_label = self.contract_label[['md5','label']]
            contract_label.columns = ['md5','labels']
            contract_label.set_index('md5',inplace=True)
            return contract_label            
    
   
def delete_and_insert_target(file, target_table, df):
    delete_mysql_by_df(target_table, df)
    mysql_insert_data(df,target_table)
    move_file(file)

def deal_detail_content(html):
    res = ''
    if html:
        sj = Selector(text=html)
        sj.xpath('//script | //noscript | //style').remove()
        content = sj.xpath('string(.)').extract_first(default='')
        content = re.sub(r'[\U00010000-\U0010ffff]', '', content)
        content = re.sub(r'\s', '', content)
        content_ = re.findall(r'\w', content)
        if content_:
            res = repr(''.join(content_))
    return res

# 必需数据是否满足
def is_full_data(task,df):
    import numpy as np
    if task == 'contract' and any(x in list(df[['notice_num','tenderee']].fillna('')) for x in ['']):
        return 0
    elif list(df[['province','city','county']].fillna(''))==['','','','','']:
        return 0
    elif task =='tender' and (any(x in list(df[["project_name","notice_num","budget","tenderee","tender_document_stime","tender_etime",'publish_time','title']].fillna('')) for x in ['']) or any(x in str(df['budget']) for x in ['元','万']) or 0<df['budget']<100 or df['budget'] != df['budget']):
        return 0
    elif task == 'bid' and (not all(x in df['labels'].split("\"") for x in ["项目名称","中标公告编号","中标金额","中标单位"]) or any(x in list(df[['publish_time','title']].fillna('')) for x in ['']) or any(x in str(df['amount']) for x in ['元','万']) or 0<df['amount']<100 or df['amount'] != df['amount'] or len(str(int(df['amount']))) > 11):
        return 0
    else:
        return 1

# 多中标单位连接
def deal_winning_bidders(labels):
    try:
        labels = eval(labels)
    except:
        return ''
    str = []
    for label in labels:
        for key,value in label.items():
            if key=="中标单位":
                str.append(value)
    return "_".join(str)

if __name__ == '__main__':
    helper = Helper()
    data_process = json.loads(open('data_process.json','r',encoding='utf-8').read())
    while True:
        files = glob.glob(DATA_PATH + '*.json')
        for file in tqdm(files):
            try:
                print(file)
                task = file.split('#')[0].split('/')[-1]
                print(task)
                # task = 'bid'
                origin_table = file.split('#')[1]
                target_table = data_process.get(origin_table).get('target')

                label_data = helper.get_label(task)
                nlp = helper.get_model(task)
        
                df = pd.read_json(file)
                if task == 'contract':
                   df.drop(['winning_bidder','amount','contract_term','agency','bid_section_name'],axis=1,inplace=True)

                # 删除更新时间
                df = df.drop(columns=['update_time'])
                
                # 提取文本
                df['result_detail'] = df['detail_content'].fillna('').apply(deal_detail_content)
                if len(df) == 0:
                    continue
                df['text'] = df['detail_content'].fillna('')
                df['text'] = df['text'].apply(p_filter_tags)
                df['md5'] = df['text'].apply(p_generate_md5)
                max_len = 10000
                df['text'] = df['text'].str[:max_len]
                # 计算标签
                data = df['text'].to_list()
                docs = nlp.pipe(data)
                labels = []

                for doc in docs:
                    label = []
                    for ent in doc.ents:
                        label.append([ent.start_char,ent.end_char,ent.label_])
                    labels.append(label)

                df['labels'] = labels
                # 替换正确标签
                df.loc[df.md5.isin(label_data.index),'labels'] = df[df.md5.isin(label_data.index)]['md5'].apply(lambda x:find_labels_by_md5(x,label_data))

                
                if all_labels_is_empty(df['labels']):
                    df = df.drop(columns=['md5','text'])
                    df['labels'] = ''
                    df = datetime_process(df,task)
                    if origin_table in ['test_tender_bid','test_tender_bid_result']:
                        df['is_full_data'] = 0
                    delete_and_insert_target(file, target_table, df)
                    if task == 'bid' and len(df) > 0:
                        df['announcement_id'] = df['id']
                        delete_win_by_df('final_winning_bidder',df,origin_table)
                    continue
                
                if task == 'tender':
                    df['money'] = df['budget']
                if task == 'bid':
                    df['money'] = df['amount']
                df['money'] = df['money'].fillna('')
                df['money'] = df['money'].apply(lambda x:clean_money(x))
                # 得到label内容
                df['labels'] = df.apply(get_labels,axis=1)
                # label根据任务做清洗
                df['clean_labels'] = df['labels'].apply(lambda x:clean_labels(x,task))
                # 根据任务，处理label之间的问题
                df['clean_labels'] = df['clean_labels'].apply(lambda x:process_labels(x,task))
                # 根据任务预处理数据，合并标签内容，有全部合并成一个或者是取最前面的数据
                df['clean_labels'] = df['clean_labels'].apply(lambda x:preprocess_save(x,task))
                # 处理填充数据，可能根据任务分列
                label_df, sub_label_df = process_save(data_process, task, origin_table, df)
                label_df = label_df[label_df.columns.drop(list(label_df.filter(regex='[^A-Za-z_\d]')))]
                # 填充label数据
                cols = list(data_process[task].values())
                for col in label_df.columns:
                    for md5,value in label_df[label_df[col].notnull()][['md5',col]].values:
                        df.loc[df.md5 == md5,col] = value
                # 填充子表
                if task == 'bid':
                    df['announcement_id'] = df['id']
                    delete_win_by_df('final_winning_bidder',df,origin_table)
                    df.drop(columns=['announcement_id'],inplace=True)
                    if len(sub_label_df) > 0:
                        mysql_insert_data(sub_label_df,'final_winning_bidder')

                # 清理df数据
                df['labels'] = df['labels'].apply(lambda x: json.dumps(x,ensure_ascii=False))
                # 设置labels最长长度1000
                # df['labels'] = df['labels'].apply(lambda x: x[:1000])
                df = df.drop(columns=['md5','text','clean_labels'])
                df = datetime_process(df,task)
                
                if task == 'bid':
                    df['winning_bidder'] = df['labels'].apply(lambda x:deal_winning_bidders(x))
                    df['amount'] = df.apply(lambda series:series['amount'] if series['money']==0 else series['money'],axis=1)
                elif task == 'tender':
                    df['budget'] = df.apply(lambda series:series['budget'] if series['money']==0 else series['money'],axis=1)
                df.drop(columns=['money'],inplace=True)

                if origin_table in ['test_tender_bid','test_tender_bid_result','test_tender_trade_result']:
                    df['is_full_data'] = df.apply(lambda x:is_full_data(task,x),axis=1)                  
                                    
                # 填写数据
                delete_and_insert_target(file, target_table, df)
            except Exception as e:
                print(e)
                file_name = file.split('/')[-1]
                os.rename(file,DATA_PATH + 'tmp/' + file_name)
                with open('/root/error_collect.txt','a') as f:
                    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f'{t}\n{file_name}\n{e}\n')
                    f.close()
        time.sleep(10)


