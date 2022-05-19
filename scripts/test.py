from unittest import result
from data_utils import *
from google_utils import *
from mysql_utils import *
from langdetect import detect

b_convert_baidu_dataset('train_dev.json',1000)