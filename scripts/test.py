
from data_utils import * 

import re
import datetime

sample = {'type':'报名结束时间','value':'2022年3月22日'}

new_data = [sample]

for sample in new_data:
    value = sample['value']
    value = re.findall(r'\d+', value)
    if sample['type'] in ['报名开始时间','报名结束时间','招标文件领取开始时间','招标文件领取结束时间','投标截止时间','开标时间']:
        if value:
            # 如果第一位不是年
            if len(value[0]) != 4:
                year = datetime.datetime.now().strftime('%Y')
                if len(value[0]) == 2 and int(value[0]) > 12:
                    value[0] = '20' + value[0]
                    break
                value.insert(0,year)
            if len(value) < 6:
                    for i in range(6 - len(value)):
                        value.append(0)
            # 全部转换成数字
            value = [int(i) for i in value]
            # 如果24点
            if value[3] == 24:
                value[3] = 23
                value[4] = 59
                value[5] = 59
            value = datetime.datetime(value[0], value[1], value[2], value[3], value[4], value[5])
            # 转换成标准时间格式
            value = value.strftime('%Y-%m-%d %H:%M:%S')
            sample['time'] = value

