

from data_utils import *


nlp = b_trf_load_model()


b_trf_label_dataset(nlp,'train_dev.json')