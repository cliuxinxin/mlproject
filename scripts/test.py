from data_utils import *
from google_utils import *
from mysql_utils import *

project_configs['tender']


b_get_target_table()




b_get_dataprocess()

document.save(r"D:\test.docx")

comapare = b_read_db_compare()

# comapare 是df
# 统计 task 的数量
comapare.groupby('task').count()

