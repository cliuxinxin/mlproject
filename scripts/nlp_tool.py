import os
import nltk

import pandas as pd
from umap import UMAP

# pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer

# 根目录
ROOT = os.path.dirname(os.path.dirname(__file__))
ASSETS = os.path.join(ROOT, 'assets/')


class Processer():
    def umap_process(self,df,filename='ready.csv'):
        # python3 -m bulk text ready.csv --keywords 'keyword1,keyword2' 
        model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        sentences = df['text']
        X =  model.encode(sentences)
        umap = UMAP()
        X_tfm = umap.fit_transform(X)
        df['x'] = X_tfm[:, 0]
        df['y'] = X_tfm[:, 1]
        df.to_csv(ASSETS + filename)


class Cleaner():
    def find_n(self,text):
        if text == '\n':
            return True
        return False


class NLPTool:
    def read_file(self,filename):
        with open(ASSETS + filename,'r') as f:
            lines = f.readlines()
        return lines


class Dataset():
    def __init__(self) -> None:
        self.df = None
        pass

    def read_csv(self,path):
        df = pd.read_csv(ASSETS + path)
        self.df = df 

    def sample(self,n):
        return self.df.sample(n)

class Base():
    def add_length(self,df,col='text'):
        df['length'] = df[col].apply(lambda x:len(x))
        return df

    def display_describe(self,df):
        return df.describe().T

    def display_descirbe_num(self,df,txt_col=['text','title']):
        return df[txt_col].describe().T

    def display_na(self,df):
        return df.isnull().sum()

    def display_length(self,df):
        df['length'].plot(kind='hist',bins=30,figsize=(8,2))

    def fill_na(self,df,col,value='unkown'):
        df[col] = df[col].fillna(value)
        return df

    def nltk_stopwords(sefl,lng='english'):
        stopwords = set(nltk.corpus.stopwords.words(lng))
        return stopwords

    def add_stopwords(self,stopwords,include_stopwords):
        stopwords = stopwords | include_stopwords
        return stopwords

    def remove_stopwords(self,stopwords,exclude_stopwords):
        stopwords -= exclude_stopwords
        return stopwords 
        

file = 'xiyouji.txt'

cleaner = Cleaner()

lines = NLPTool().read_file(file)
lines = [line for line in lines if cleaner.find_n(line)!=True]

df = pd.DataFrame(lines)
df.columns = ['text']
Processer().umap_process(df,'xiyouji.csv')



