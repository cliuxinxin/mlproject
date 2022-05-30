from data_utils import *


files = glob.glob(DATA_PATH + '*.json')
file = files[0]

df = pd.read_json(file)

df.columns

# 找出detail_content最长的数据
df['length'] = df.apply(lambda x: len(x['detail_content']), axis=1)
# 找出detail_content最短的数据
df['length'].max()

# 找出detail_content最长的数据
text = df[df['length'] == df['length'].max()]['id'].values[0]

len(text)

test = p_filter_tags(text)

test = html_to_text(text)
len(test)
len(text)


