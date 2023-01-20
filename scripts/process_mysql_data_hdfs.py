from tqdm import tqdm
from data_utils import *
from mysql_utils import *
from data_clean_new import clean_manager
from process_mysql_data_new import *



def deal_big_data(files,origin_table):
    helper = Helper()
    data_process = json.loads(open('data_process.json','r',encoding='utf-8').read())
    for file in tqdm(files):
        # 判断是否为空
        if len(file) == 0:
            continue
        df=json.loads(file)
        df=pd.DataFrame.from_dict(df,orient='index').T
        origin_table = origin_table
        task=data_process.get(origin_table).get('task')
        target_table = data_process.get(origin_table).get('target')
        label_data = helper.get_label(task)
        nlp = helper.get_model(task)
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
            # delete_and_insert_target(file, target_table, df)
            delete_mysql_by_df(target_table, df)
            mysql_insert_data(df,target_table)
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
        if task == 'bid' and len(sub_label_df) > 0:
            delete_win_by_df('final_winning_bidder',sub_label_df,origin_table)
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
        # delete_and_insert_target(file, target_table, df)
        delete_mysql_by_df(target_table, df)
        mysql_insert_data(df,target_table)


if __name__ == '__main__':
    pass