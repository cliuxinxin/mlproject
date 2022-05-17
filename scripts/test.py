from data_utils import *
from google_utils import *

df1 = pd.read_excel(ASSETS_PATH + '1.xlsx')
df2 = pd.read_excel(ASSETS_PATH + '2.xlsx')
df3 = pd.read_excel(ASSETS_PATH + '3.xlsx')


df1.columns = [x.replace('_ - ', ' ') for x in df1.columns]
df2.columns = [x.replace('_ - ', ' ') for x in df2.columns]
df3.columns = [x.replace('_ - ', ' ') for x in df3.columns]

df1.columns = [x.strip() for x in df1.columns]
df2.columns = [x.strip() for x in df2.columns]
df3.columns = [x.strip() for x in df3.columns]



print (len(df1))
print (len(df2))
print (len(df3))

df = pd.concat([df1, df2, df3])

df[df['is_human_correct'] == 'Y']

# is_human_correct 大写 Y 就是 True
df = df[df['is_human_correct'].str.upper() == 'Y']

# 设定排重列
md5_columns = ['task','md5','human_start','human_end','human_label','ai_start','ai_end','ai_label','label_type']

# md5 列组合生成md5值
df['label_md5'] = df[md5_columns].apply(lambda x: '_'.join(x.astype(str)), axis=1).apply(p_generate_md5)

# 根据 label_md5

