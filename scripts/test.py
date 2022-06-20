from data_utils import *
from mysql_utils import *
from data_clean_new import clean_manager

import rarfile

import pandas as pd

path = ASSETS_PATH + 'dorab/*'

# 找出path和子目录下所有的jpg文件
files = glob.glob(path)


