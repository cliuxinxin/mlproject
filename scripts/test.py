from data_utils import *
from mysql_utils import *
from data_clean_new import clean_manager

import rarfile

import pandas as pd

files = glob.glob(ASSETS_PATH + 'dora_b/*')

files
file = files[0]
file

# 解压rar文件
def unrar(file):
    rar_file = rarfile.RarFile(file)
    rar_file.extractall(ASSETS_PATH + 'dora_b/')
    rar_file.close()

unrar(file)