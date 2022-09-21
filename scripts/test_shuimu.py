import pandas as pd
import re

ASSETS_PATH = '../assets/'

shuimu = ASSETS_PATH + 'test.xlsx'

df = pd.read_excel(shuimu)
df.columns = ['text']

data = df['text'].tolist()

new_data = []

for entry in data:
    # 如果为nan，跳过
    if pd.isna(entry):
        continue
    # 将\xa0替换为空格
    entry = entry.replace('\xa0', ' ')
    # 提取发信人
    sender = re.findall(r'发信人: (.+?),', entry)[0]
    # 提取内容
    content = re.findall(r'站内(.+?)--', entry)[0]
    # 去掉空格
    content = content.strip()
    new_entry = {
        'sender': sender,
        'content': content
    }
    new_data.append(new_entry)

df = pd.DataFrame(new_data)

# 统计发信人
df['sender'].value_counts()

df.to_csv(ASSETS_PATH + 'test.csv', index=False)