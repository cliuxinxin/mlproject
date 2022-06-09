from data_utils import *
from mysql_utils import *
from data_clean_new import clean_manager


process = b_get_dataprocess()

for entry in process:
    new = {}
    new[entry['origin_table']] = {
        'target':entry['target_table'],
        'task':entry['task']
    }
    print(new)

