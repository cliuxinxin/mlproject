from data_utils import *

data = b_read_dataset('t.json')

from transformers import AutoModelForTokenClassification, AutoTokenizer,AutoModel
import torch

model_checkpoint = "hfl/chinese-roberta-wwm-ext"
model = AutoModel.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

sequence = [data[0]['text']]    

inputs = tokenizer(sequence, return_tensors="pt")

tokens = inputs.tokens()

outputs = model(**inputs).logits

predictions = torch.argmax(outputs, dim=2)


inputs