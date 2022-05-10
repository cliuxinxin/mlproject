
from tqdm import tqdm
from data_utils import *
from mysql_utils import *
import argparse
import glob

class PoolCorpus(object):

    def __init__(self,task,file):
        df = pd.read_json(file)
        cols = df.columns.to_list()
        std_labels = b_read_db_labels(task)
        std_labels['col_idx'] = std_labels['col'].apply(lambda x: cols.index(x))
        self.html_col = project_configs[task]['col']
        self.task = task
        self.df = df
        self.std_labels = std_labels
        self.nlp = b_load_best_model(task)

    def add(self, i):
        text = self.df.iloc[i][self.html_col]
        id = self.df.iloc[i]['id']
        text = p_filter_tags(text)
        doc = self.nlp(text)
        labels = []
        for ent in doc.ents:
          label = {}
          label['ent.label_'] = ent.text
          col = self.std_labels[self.std_labels['label'] == ent.label_].iloc[0]['col_idx']
          self.df.iloc[i,col] = ent.text
          labels.append(label)
        self.df.iloc[i]['labels'] = labels
        mysql_delete_data_by_id(id,self.task)

    def get(self):
        return self.df

def get_parser():
    parser = argparse.ArgumentParser(description="Process data and insert to mysql")
    return parser



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    files = glob.glob(DATA_PATH + '*.json')

    BaseManager.register('PoolCorpus', PoolCorpus)

    for file in tqdm(files):
        task = file.split('_')[0].split('/')[-1]
        with BaseManager() as manager:
            corpus = manager.PoolCorpus(task,file)

            with Pool() as pool:
                pool.map(corpus.add, (i for i in range(len(corpus.get()))))

            data = corpus.get()

        mysql_insert_data(data,task)
        file_name = file.split('/')[-1]
        os.rename(file,DATA_PATH + 'processed/' + file_name)

