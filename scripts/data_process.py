from data_utils import *

b_doccano_dataset_label_view('train_dev.json',['预算'],6)

db = b_extrct_data_from_db_basic('tender')

b_doccano_cat_data(db,100,['总投资','总投资额','项目规模','合同估算价','本项目投资','本项目投资','建设规模','招标控制价','工程投资','工程标的','预算'],6)
