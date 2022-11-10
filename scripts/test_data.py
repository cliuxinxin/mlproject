from refac import *
from refine_utils import *

def clean(text):
    text = text.replace(' ','')
    text = text.replace('\n','')
    text = text.replace('\r','')
    text = text.replace('\t','')
    text = text.replace('\xa0','')
    text = text.replace('\u3000','')
    text = text.replace('\u2002','')
    return text

df = pd.read_excel(assets_path('test.xlsx'))

data = []
# 循环df中的内容
for index, row in df.iterrows():
    entry = row['字段1']
    entry = entry.split('【')
    for one_entry in entry:
        if "】" in one_entry:
            one_entry = one_entry.split('】')
            row[clean(one_entry[0])] = one_entry[1]
    data.append(row)

df_new = pd.DataFrame(data)

df_new.to_csv(assets_path('test.csv'),index=False)

refine = OpenRefine('http://47.108.118.95:3333')

refine.upload_path(assets_path('test.csv'))

data[0]