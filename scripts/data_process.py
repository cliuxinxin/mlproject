from data_utils import *

# b_doccano_delete_project(1)
# b_doccano_dataset_label_view('train_dev.json',['项目招标编号'],1)

# b_doccano_train_dev_update()

# train_dev = b_read_dataset('train_dev.json')
# len(train_dev)

# t1 = time.time()
# b_label_dataset_mult('train_dev.json',50)
# t2 = time.time()
# t2-t1

from multiprocessing import Pool,Queue
import os, time, random


def run_nlp(sample,q):
    print('子流程开始')
    nlp = b_load_best_model()
    doc = nlp(sample['data'])
    label = [[ent.char_start,ent.char_end,ent.label_] for ent in doc.ents]
    sample['mlabel'] = label
    q.put(sample)
    print('子流程结束')

train = b_read_dataset('train.json')
test = train[:10]
data = []
t1 = time.time()
p = Pool()
q = Queue()
for sample in test:
    p.apply_async(run_nlp, args=(sample,q))
p.close()
p.join()
data = [q.get() for i in range(q.qsize())]
t2 = time.time()
t2 - t1
