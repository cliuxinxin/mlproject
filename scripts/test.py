from unittest import result
from data_utils import *
from google_utils import *
from mysql_utils import *
from langdetect import detect


with open(ASSETS_PATH + '1','r') as f:
    text = f.readlines()

nlp = b_load_best_model('bid')

doc = nlp(train[242]['data'])

for ent in doc.ents:
    print(ent.text,ent.label_)

doc.ents

train = b_read_dataset('train_dev.json')

text = train[242]['data']
text = text.replace('\n',' ')
text = text.replace('\t',' ')
text = text.replace('\r',' ')
with open(ASSETS_PATH + '1','w') as f:
    f.write(text)

for label in train[242]['label']:
    text = train[242]['data']
    start = label[0]
    end = label[1]
    label_ = label[2]
    print(text[start:end],label_)
