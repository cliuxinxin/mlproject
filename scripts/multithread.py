import threading
import time
import queue
from data_utils import * 

# 下面来通过多线程来处理Queue里面的任务：
def work(q,nlp,data):
    while True:
        if q.empty():
            return
        else:
            sample = q.get()
            doc = nlp(sample['text'])
            label = [[ent.start,ent.end,ent.label_] for ent in doc.ents]
            sample['mlabel'] = label
            data.append(sample)

def main():
    train = b_read_dataset('train_dev.json')
    nlp = b_load_best_model()
    data = []
    q = queue.Queue()
    for sample in train:
        q.put(sample)
    thread_num = 10
    threads = []
    for i in range(thread_num):
        t = threading.Thread(target=work, args=(q,nlp,data))
        # args需要输出的是一个元组，如果只有一个参数，后面加，表示元组，否则会报错
        threads.append(t)
    # 创建5个线程
    for i in range(thread_num):
        threads[i].start()
    for i in range(thread_num):
        threads[i].join()
    # b_save_list_datasets(data,'train_dev.json')

if __name__ == "__main__":
    start = time.time()
    main()
    print('耗时：', time.time() - start)