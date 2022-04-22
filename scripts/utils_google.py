from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

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
    gfile = drive.CreateFile({'parents': [{'id': parent_id}]})
    # Read file and set it as a content of this instance.
    gfile.SetContentFile(upload_file)
    gfile.Upload() # Upload the file.


def gdrive_find_file(file_name):
    """
    查找某个文件或者文件夹
    """
    file_list = drive.ListFile({'q': " title='%s' " % file_name }).GetList()
    return file_list[0]


# List files in Google Drive
def gdrive_list_foler(file_id):
    """
    罗列某个文件夹下的文件
    """
    fileList = drive.ListFile({'q': "'%s' in parents and trashed=false" % file_id}).GetList()
    folders = []
    for file1 in fileList:
        folders.append([file1['title'], file1['id']])
    return folders


# 递归下载文件夹
def gdrive_download_folder(folder_name,folder_id):
    """
    下载某个文件夹下的所有文件和文件夹
    """
    os.mkdir(folder_name)
    file_list = drive.ListFile({'q': "'%s' in parents and trashed=false" % folder_id}).GetList()
    for file1 in file_list:
        if file1['mimeType'] == 'application/vnd.google-apps.folder':
            gdrive_download_folder(file1['id'], os.path.join(folder_name, file1['title']))
        else:
            print('title: %s, id: %s' % (file1['title'], file1['id']))
            gfile = drive.CreateFile({'id': file1['id']})
            gfile.GetContentFile(folder_name + '/' + file1['title'])


