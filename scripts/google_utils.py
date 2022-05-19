from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import shutil

# Rename the downloaded JSON file to client_secrets.json
# The client_secrets.json file needs to be in the same directory as the script.

gauth = GoogleAuth()
drive = GoogleDrive(gauth)

# List files in Google Drive
def gdrive_list_root():
    """
    取得根目录下的文件列表
    """
    fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    return fileList


# Upload files to your Google Drive
def gdrive_upload(upload_file,parent_id):
    """
    上传文件
    """
    file_name = os.path.basename(upload_file)
    gfile = drive.CreateFile({'parents': [{'id': parent_id}]})
    # Read file and set it as a content of this instance.
    gfile['title'] = file_name
    gfile.SetContentFile(upload_file)
    gfile.Upload() # Upload the file.


def gdrive_find_file(file_name):
    """
    查找某个文件或者文件夹
    """
    file_list = drive.ListFile({'q': " title='%s' " % file_name }).GetList()
    return file_list[0]



def gdrive_list_foler(file_id):
    """
    罗列某个文件夹下的文件
    """
    fileList = drive.ListFile({'q': "'%s' in parents and trashed=false" % file_id}).GetList()
    folders = []
    for file1 in fileList:
        folders.append([file1['title'], file1['id']])
    return folders



def gdrive_download_folder(file_id, download_path):
    """
    下载文件夹
    """
    file_list = drive.ListFile({'q': "'%s' in parents and trashed=false" % file_id}).GetList()
    for file1 in file_list:
        if file1['mimeType'] == 'application/vnd.google-apps.folder':
            # 如果是文件夹，则递归下载
            os.makedirs(download_path + '/' + file1['title'], exist_ok=True)
            gdrive_download_folder(file1['id'], download_path + '/' + file1['title'])
        else:
            # 如果路径不存在，则创建路径
            os.makedirs(os.path.dirname(download_path + '/' + file1['title']), exist_ok=True)
            gdrive_download_file(file1['id'], download_path + '/' + file1['title'])


def gdrive_download_file(file_id, download_path):
    """
    下载文件
    """
    file_list = drive.CreateFile({'id': file_id})
    file_list.GetContentFile(download_path)


def gdrvie_find_file_under_folder(name,folder_id):
    """
    在folder_id下，查找到title名叫name的文件
    
    """
    file_list = drive.ListFile({'q': "'%s' in parents and trashed=false and title='%s'" % (folder_id,name)}).GetList()
    return file_list[0]

def gdrive_del_and_upload(file,folder_id):
    """
    找到并且删除某个folder_id下面的文件，并且上传新的文件
    """
    file_name = os.path.basename(file)
    try:
        gfile = gdrvie_find_file_under_folder(file_name,folder_id)
        gfile.Delete()
    finally:
        gdrive_upload(file,folder_id)

def gdrive_upload_train_dev():
    """
    上传本地的train.json,dev.json,train_dev.json到dojo的assets文件夹下
    """
    gdrive_assets_id = '1C88ng_CyJW_zzD1Bp7Sr_A9OCs128sbr'
    train_file = '../assets/train.json'
    dev_file = '../assets/dev.json'
    train_dev_file = '../assets/train_dev.json'

    gdrive_del_and_upload(train_file,gdrive_assets_id)
    gdrive_del_and_upload(dev_file,gdrive_assets_id)
    gdrive_del_and_upload(train_dev_file,gdrive_assets_id)

def gdrive_upload_cats_train_dev():
    """
    上传本地的train.json,dev.json到dojo的assets文件夹下
    """
    gdrive_assets_id = '1C88ng_CyJW_zzD1Bp7Sr_A9OCs128sbr'
    train_file = '../assets/train_cats.json'
    dev_file = '../assets/dev_cats.json'

    gdrive_del_and_upload(train_file,gdrive_assets_id)
    gdrive_del_and_upload(dev_file,gdrive_assets_id)

def gdrive_find_and_download(folder_id,file_name):
    """
    在folder_id下，查找到文件并且下载回来
    
    """
    gdrvie_training_folder_id = folder_id

    file = gdrvie_find_file_under_folder(file_name,gdrvie_training_folder_id)

    # 如果是文件，就下载文件
    if file['mimeType'] != 'application/vnd.google-apps.folder':
        gdrive_download_file(file['id'],file_name)
    else:
        gdrive_download_folder(file['id'], file['title'])


def gdrive_find_file_under_folder_download_move(parent_id,file,target_path):
    """
    在parent_id下，查找到file并且下载回来，并且移到target_path下面

    """
    gdrive_find_and_download(parent_id,file)
 
    gdrive_del_move(file,target_path+file)

def gdrive_del_move(orig_path,target_path):
    """
    将orig_path文件夹移到target_path下面，如果有，则强制覆盖
    """
    # 如果目标文件夹存在
    if os.path.exists(target_path):
        # 如果是文件，就删除文件
        if os.path.isfile(target_path):
            os.remove(target_path)
        else:
            shutil.rmtree(target_path)
    shutil.move(orig_path, target_path)


def gdrive_download_best_model_cats():
    """
    把cats训练最好的模型下载到本地的training目录下面
    
    """
    parent_id = '13341aaueIdJUpPQozOsXlTMkZ6KDdflT'
    file = 'model-best'
    target_path = '../training/cats/' 

    gdrive_find_file_under_folder_download_move(parent_id,file,target_path)


def gdrive_download_best_model(task):
    """
    根据任务把ner训练最好的模型下载到本地的training目录下面
    
    """
    folder_id = '1D4jgGQhuXbOfA15hZJAYaNgVKiisXNsW'
    parent_id = gdrvie_find_file_under_folder(task,folder_id)['id']
    file = 'model-best'
    target_path = '../training/{}/'.format(task)

    gdrive_find_file_under_folder_download_move(parent_id,file,target_path)

def gdrive_download_metrics():
    """
    下载metrics到本地
    """
    parent_id = '1D4jgGQhuXbOfA15hZJAYaNgVKiisXNsW'
    file = 'metrics'
    target_path = '../training/'

    gdrive_find_file_under_folder_download_move(parent_id,file,target_path)

def gdrive_download_labeled_data():
    """
    下载标注好的数据
    """
    parent_id = '1C88ng_CyJW_zzD1Bp7Sr_A9OCs128sbr'
    file = 'train_dev_label.json'
    target_path = '../assets/'

    gdrive_find_file_under_folder_download_move(parent_id,file,target_path)











