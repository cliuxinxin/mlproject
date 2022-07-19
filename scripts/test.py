from data_utils import *

data = b_read_dataset('relation.jsonl')

data[0]

example = b_read_dataset('rel_example.jsonl')

example[3]

# 找到所有的keys
keys = set()
for d in data:
    keys.update(d.keys())
print(keys)

def b_display_all_keys(data):
    keys = set()
    for entry in data:
        keys.update(entry.keys())
    print(keys)

b_display_all_keys(example)

# 找到一个有relations的entry，打印出来
for entry in example:
    if 'relations' in entry:
        print(entry)
        break

path = DATA_PATH + 'train.spacy'

from spacy.tokens import DocBin,Doc

doc_bin = DocBin().from_disk(path=path)
nlp = spacy.blank("zh")
doc_bin = DocBin().from_bytes(bytesdata)
docs = list(doc_bin.get_docs(nlp.vocab))
Doc.set_extension("rel", default=None)
docs[].ents

docs[3]._.rel
docs[3].ents[0].start
docs[3].ents[1].start

# 核验同一个标注数据中，id木有重复的
# 核验同一个标注中，relations 终端的from id 和 to id 要在id中存在

for entry in data:
    # id不能重复
    id_set = {}
    for one_label in entry['entities']:
        id = one_label['id']
        if id in id_set:
            print(id)
            print(entry)
            exit()
        id_set[id] = 1

    # relations 终端的from id 和 to id 要在id中存在
    for one_relation in entry['relations']:
        from_id = one_relation['from_id']
        to_id = one_relation['to_id']
        if from_id not in id_set:
            print(from_id)
            print(entry)
            exit()
        if to_id not in id_set:
            print(to_id)
            print(entry)
            exit()
