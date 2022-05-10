from data_utils import * 

task = 'bid'

labels = b_read_db_labels(task)
labels = labels['label'].to_list()

