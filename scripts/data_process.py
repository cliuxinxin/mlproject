from data_utils import *
from google_utils import *

# 定义任务
# task = 'tender'
task = 'bid'
method = 'process'
# method = 'thread'

# ---- DOJO 训练完成模型 --------
# 下载最好的模型
gdrive_download_best_model()

# 从doccano上下载最新的train和dev，并且合并为train_dev
b_doccano_train_dev_update(task)

# 迭代解决空格问题
b_remove_invalid_label('train_dev.json')
b_generate_compare_refine(task,'train_dev.json','train_dev_remove.json')

# 标注
t1 = time.time()
if method == 'process':
    b_label_dataset_multprocess('train_dev.json')
else:
    b_label_dataset_mult('train_dev.json',20)

t2 = time.time()
t2 - t1

# 根据标注和AI的情况生成cats数据集
b_generate_cats_datasets_by_compare('train_dev.json','train_dev_label.json')

# ------ 线下通过refine对比数据 -------------
# 根据差异生成refine对比表
b_generate_compare_refine(task,'train_dev.json','train_dev_label.json')

# 上传cats数据集
gdrive_upload_cats_train_dev()

# ----- DOJO 训练cats模型 --------------
# 下载生成的cats模型
gdrive_download_best_model_cats()

# 选择数据
data = b_select_data_by_model(task,300)

# 标注分割和上传
b_devide_data_import(data,task,method,20)








