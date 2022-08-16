from tqdm import tqdm
from data_utils import *
from mysql_utils import *
    
sql = 'select distinct source_website_name from final_other_tender_bid where project_name is null limit 10'
sql = "select * from final_other_tender_bid where project_name is null and source_website_name='宝华智慧招标共享平台' limit 10"

df = mysql_select_df(sql)

task = 'tender'
source = 'final_other_tender_bid'
file_name = task + '#' + source + '#' + str(int(time.time()*100000))
df[:5].to_json(DATA_PATH + file_name + '.json')


col = 'detail_content'
source = 'source_website_address'

df2 = df[5:]

data=df2[[col,source,'id']]
a=p_process_df(data,task)
b_save_df_datasets(a,"one_data.json")
# 加入标签
b_label_dataset_mult(task,'one_data.json',20)

b_doccano_upload_by_task('one_data_label.json',task,'train')

b_doccano_train_dev_update(task)

df = pd.read_json(DATA_PATH + 'tender#test_other_tender_bid#165940305577271.json')

df.id.values

train_dev = b_read_dataset('train_dev.json')

task = 'tender'

data = b_read_dataset('train_dev.json')

train_project_id = project_configs[task]['train']
dev_project_id = project_configs[task]['dev']

label = '预算'
keyword = '招标文件费'

new_data = []
for entry in data:
    for human_start,human_end,label_ in entry['label']:
        new_entry = {}
        new_entry['task'] = entry['task']
        md5 = entry['md5']
        new_entry['md5'] = md5
        text = entry['text']
        human_label = text[human_start:human_end]
        new_entry['human_start'] = human_start
        new_entry['human_end'] = human_end
        new_entry['label_type'] = label_
        new_entry['human_label'] = human_label
        new_entry['label_len'] = len(human_label)
        new_entry['is_human_correct'] = ''
        try:
            new_entry['url'] = entry['data_source'] + "#:~:text=" + str(human_label)
        except:
            new_entry['url'] = ''
        dataset = train_project_id if entry['dataset'] == task + '_train' else dev_project_id
        new_entry['doccano_url'] = 'http://47.108.118.95:18000/projects/{}/sequence-labeling?page=1&q={}'.format(dataset,md5) 
        if label_ == label and text.find(keyword) != -1:
            new_data.append(new_entry)
        

# 日期作为file_name
file_name = str(int(time.time()*100000))
b_save_list_datasets(new_data,f'{file_name}.json')

data = b_read_dataset('dev.json')

df = pd.DataFrame(data)

# 如果 data_source 为空，则用 source 字段 填充
df.loc[df.data_source.isnull(),'data_source'] = df.loc[df.data_source.isnull(),'source']

# 去掉 source 字段
df.drop(columns=['source'],inplace=True)

b_save_df_datasets(df,'dev.json')

b_doccano_upload_by_task('dev.json','tender','dev')

task = 'tender'

b_sample_label_data(task)







