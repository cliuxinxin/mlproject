import os
import re
import pandas as pd

ASSETS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')

cov_path = os.path.join(ASSETS_PATH,'cov')

cov_files = os.listdir(cov_path)

pattern = re.compile(r'居住于(锦江区|青羊区|双流区|龙泉驿区|新都区|金牛区|温江区|高新区|成华区|天府新区|武侯区)(.*?)，系(.*?)。')

new_data = []

for file in cov_files:
    data = open(os.path.join(cov_path,file),'r').read()
    data = data.split('\n')
    for text in data:
        result = pattern.findall(text)
        if result:
            entry = {}
            entry['日期'] = file
            entry['区域'] = result[0][0]
            entry['小区'] = result[0][1]
            entry['情况'] = result[0][2]
            new_data.append(entry)

df = pd.DataFrame(new_data)

# 根据日期和情况统计人数
df_group = df.groupby(['日期','情况'])['小区'].count().reset_index()
df_group[df_group.情况=='主动就诊发现']
df_group[df_group.情况.str.contains('愿检尽检')]

# 根据 区域 日期 统计人数
df_group = df.groupby(['区域','日期'])['小区'].count().reset_index()

df[df.区域.str.contains('金牛')]
df[df.区域.str.contains('龙泉')]
df[df.区域.str.contains('武侯')]
df[df.区域.str.contains('锦江') & df.日期.str.contains('2022-8-31')]



df[df.日期.str.contains('2022-8-30')&(df.情况.str.contains('愿检')|df.情况.str.contains('主动'))].sort_values(by='区域')

df[df.区域.str.contains('高新')].sort_values(by=['日期','小区'])
df[df.区域.str.contains('龙泉')].sort_values(by=['日期','小区'])