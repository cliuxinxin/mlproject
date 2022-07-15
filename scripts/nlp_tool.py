import os
import nltk
import re

import pandas as pd
from requests import head
from umap import UMAP
import numpy

# pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer
from websockets import Data

# 根目录
ROOT = os.path.dirname(os.path.dirname(__file__))
ASSETS = os.path.join(ROOT, 'assets/')

file = 'xiyouji.txt'

df = pd.read_csv(ASSETS + file,header=None)

df.columns = ['text']

sentences = df['text']

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

X =  model.encode(sentences)

numpy.savetxt(ASSETS + "xiyouji.tsv", X, delimiter='\t')

df.to_csv(ASSETS + "xiyouji_meta.csv", index=False,header=False)


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
    def __init__(self,path) -> None:
        self.df = self.read_csv(path)
        pass

    def read_csv(self,path):
        df = pd.read_csv(ASSETS + path)
        return df

    def sample(self,n):
        return self.df.sample(n)

class Base():
    def add_length(self,df,col='text'):
        df['length'] = df[col].apply(lambda x:len(x))
        return df

    def tokenize(sefl,text):
        return re.findall(r'[\w-]*\p{L}[\w-]*', text) 

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

    def exclude_stopwords(self,stopwords,exclude_stopwords):
        stopwords -= exclude_stopwords
        return stopwords 
    
    def remove_stopwords(self,tokens,stopwords):
        return [token for token in tokens if token not in stopwords]

    def prepare_pipeline(self,text,pipeline=[str.lower,tokenize,remove_stopwords]):
        tokens = text
        for transform in pipeline:
            tokens = transform(tokens)
        return tokens

    def pipeline_df(self,df):
        df['tokens'] = df['text'].apply(lambda x:self.prepare_pipeline(x))
        return df



