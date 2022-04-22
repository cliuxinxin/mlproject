from data_utils import *

b_doccano_delete_project(1)
b_doccano_dataset_label_view('train_dev.json',['项目招标编号'],1)

db = b_extrct_data_from_db_basic('tender')

b_doccano_cat_data(db,100,['项目招标编号','项目编号','招标编号','招标项目编号','项目代码','标段编号','标段编号为'],1)


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Rename the downloaded JSON file to client_secrets.json
# The client_secrets.json file needs to be in the same directory as the script.
gauth = GoogleAuth()
drive = GoogleDrive(gauth)

# List files in Google Drive
fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
for file1 in file_list:
  print('title: %s, id: %s' % (file1['title'], file1['id']))



