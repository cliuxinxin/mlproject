import requests
from data_utils import b_read_dataset
from scripts.data_utils import ASSETS_PATH

ASSETS_PATH  = '../assets/'

# refine url


file_name = 'compare_results.json'
file = ASSETS_PATH + file_name
file


def get_csrf_token():
    url = 'http://127.0.0.1:3333/command/core/get-csrf-token'
    r = requests.get(url)
    return r.json()['token']

# 创建project
def create_project(project_name,file_name):
    url = '127.0.0.1:3333'
    url = 'http://' + url + '/command/core/create-project-from-upload'
    file = ASSETS_PATH + file_name
    data = {
        'project_name': project_name,
        'project_file': file_name,
        'format':'ext/line-based/*sv',
        'options': {
            'project_file' : {
                'file': open(file, 'rb'),
                'filename': file_name
            }
         }
        }
    params = {
        'token': get_csrf_token()
    }
    r = requests.get(url, data=data, params=params)
    return r.json()

create_project('test','test.csv')

test = b_read_dataset('compare_results.json')
import json
# 保存为json文件
with open('test.json', 'w', encoding='utf-8') as f:
    json.dump(test, f, ensure_ascii=False)

import pandas as pd
test = pd.DataFrame(test)
test.to_csv(ASSETS_PATH + 'test.csv',index=False)


