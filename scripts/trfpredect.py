from data_utils import *

nlp = b_trf_load_model()

b_trf_label_dataset(nlp,'train_dev.json')

b_eavl_dataset('train_dev.json','train_dev_trf_label.json')