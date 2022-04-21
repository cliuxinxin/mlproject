from data_utils import *
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments)

from transformers import pipeline

path = 'bert-base-chinese-finetuned-ner/checkpoint-7000'

train_dev = b_read_dataset('train_dev.json')

text = train_dev[15]['data']

l_text = list(text)


tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForTokenClassification.from_pretrained(path)

nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='simple')
ents = nlp(text)

for ent in ents:
    if ent['entity_group'] != 'LABEL_0':
        print(ent)
