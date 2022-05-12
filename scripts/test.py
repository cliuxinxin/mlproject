from data_utils import *
from data_clean import *
from datetime import datetime

task = 'bid'

b_gpu_label(task,'train_dev.json')

# 去掉 data_source 为na的数据
db = db[~db['data_source'].isnull()]

b_save_db_basic(db)


from refine_utils import * 

file = '../assets/compare_results.json'

create(file)

