
import numpy as np
from datasets import load_dataset, load_metric
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments)
import torch
from torch import nn
from transformers import Trainer

from data_utils import *

# 对齐label
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["data"], truncation=True, is_split_into_words=True,padding='max_length', max_length=512)

    labels = []
    for i, label in enumerate(examples["label"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

model_checkpoint = "hfl/chinese-roberta-wwm-ext"
task = 'ner'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

bio_labels = b_bio_labels_generate_from('labels.txt')

datasets = load_dataset('json', data_files= {'train': ASSETS_PATH + 'train_trf_maxlen.json', 'dev': ASSETS_PATH + 'dev_trf_maxlen.json'})

label_all_tokens = True

tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
# 去掉 data,label
tokenized_datasets = tokenized_datasets.remove_columns(["data", "label"])

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(bio_labels))

model_name = model_checkpoint.split("/")[-1]

batch_size = 8

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "steps",
    eval_steps = 10000,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=180,
    weight_decay=0.01,
    save_strategy = 'steps',
    save_steps = 10000,
    load_best_model_at_end = True
)


metric = load_metric("seqeval")

data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [bio_labels[p] for (p, l) in zip(prediction, label) if l != -100 ]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [bio_labels[l] for (p, l) in zip(prediction, label) if l != -100 ]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

weight = [ 10 ] * len(bio_labels)
weight[0] = 1
weight = torch.tensor(weight, dtype=torch.float)
# 放到gpu
weight = weight.cuda()

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['dev'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model('trf_model')






