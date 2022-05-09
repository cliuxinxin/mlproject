from data_utils import *

task = 'tender'
b_doccano_update_train_dev(task)

train = b_read_dataset('train.json')
dev = b_read_dataset('dev.json')

b_doccano_upload_by_task('train.json',task,'train')
b_doccano_upload_by_task('dev.json',task,'dev')