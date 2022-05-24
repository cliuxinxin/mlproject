from data_utils import *
from google_utils import *
from mysql_utils import *

data = b_read_dataset('train_dev_label.json')

data = pd.DataFrame(data)

data['task'] = 'tender'
# 去掉file_name
data = data.drop(['file_name'],axis=1)
# 去掉 空白 
data = data.drop([''],axis=1)

b_save_df_datasets(data,'train_dev_label.json')

task = 'bid'
sql = 'select * from bid_test where project_name is null order by rand() limit 1'

df = mysql_select_df(sql)

labels = json.loads(df.labels[0])

source = project_configs[task]['col']



from gne import GeneralNewsExtractor

html = df[source][0]

extractor = GeneralNewsExtractor()
result = extractor.extract(html)
print(result)

nlp = b_load_best_model(task)
doc = nlp(result['content'])
for ent in doc.ents:
    print(ent.text, ent.label_)

sour = project_configs[task]['source']
df[sour][0]

