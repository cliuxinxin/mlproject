import json
import os
import zipfile
from doccano_client import DoccanoClient

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_PATH = os.path.join(PROJECT_PATH, 'assets')

def get_configs(file_name='configs.json'):
    """
    读取配置文件
    """
    configs = json.load(open(file_name))
    return configs

def get_config(configs,config_name):
    """
    读取具体的配置
    """
    config = configs[config_name]
    return config

def get_doccano_client():
    """
    获取doccano客户端
    """
    # 检查是否有
    configs = get_configs()
    config_doccano = get_config(configs,'doccano')
    client = DoccanoClient(config_doccano['url'])
    client.login(username=config_doccano['user'], password=config_doccano['password'])
    return client

def b_doccano_project_export(client,project_id,file_name):
    """
    导出项目，并且保存到ASSETS文件夹下面
    """
    file_name = client.download(project_id,'JSONL')
    zipfile.ZipFile(file_name).extractall(path=ASSETS_PATH)
    os.rename(os.path.join(ASSETS_PATH,'all.jsonl'),os.path.join(ASSETS_PATH,file_name))
    os.remove(file_name)

def d_read_json(path) -> list:
    """
    读取jsonl文件
    """
    data = []
    with open(path, 'r',encoding='utf8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def b_read_dataset(file):
    """
    将ASSETS文件夹下的文件读取出来
    """
    data = d_read_json(os.path.join(ASSETS_PATH,file))
    return data


def b_save_dataset(data,path,is_df='N'):
    """
    把数据集保存到ASSETS文件夹下
    """
    if is_df == 'Y':
        data = data.to_dict('records')
    with open(path,'w',encoding='utf-8') as f:
        for entry in data:
            json.dump(entry,f,ensure_ascii=False)
            f.write('\n')

def b_save_dataset(data,file,is_df='N'):
    """
    把数据集保存到ASSETS文件夹下
    """
    b_save_dataset(data,os.path.join(ASSETS_PATH,file),is_df)




# client = get_doccano_client()
# b_doccano_project_export(client)
data = b_read_dataset('data.json')
b_save_dataset(data,'data_1.json')


