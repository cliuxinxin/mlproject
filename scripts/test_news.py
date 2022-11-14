from refac import *
from gne import GeneralNewsExtractor


def get_htmls():
    """
    从数据库中获取htmls文件目录
    """
    html_path = os.path.join(ASSETS_PATH,'htmls')
    htmls = os.listdir(html_path)
    htmls = [os.path.join(html_path,html) for html in htmls if html != '.DS_Store']
    return htmls

def get_htmls_content(htmls):
    """
    从htmls文件中获取content放到data
    """
    extractor = GeneralNewsExtractor()
    data = []
    for html in htmls:
        with open(html,'r',encoding='utf-8') as f:
            text = f.read()
        try:
            result = extractor.extract(text)
            data.append({
                'text':result['title'] + '\n' + result['content'],
                'author':result['author'],
                'name':html,
                'date':result['publish_time']
        })
        except:
            pass
    return data

def get_new_data(data,name='train.json'):
    """
    对比htmls文件夹里面的文件和doccano里面的数据，取出doccano里面没有的。
    """
    doc_data = b_read_dataset(name) 

    name_data = [doc['name'] for doc in doc_data]

    new_data = []

    for entry in data:
        if entry['name'] not in name_data:
            new_data.append(entry)
    return new_data

def get_doc_id(name='train.json'):
    """
    返回实体和关系的ID
    """
    doc_data = b_read_dataset(name)
    ent_ids = []
    rel_ids = []
    for sample in doc_data:
        for ent in sample['entities']:
            ent_ids.append(ent['id'])
        for rel in sample['relations']:
            rel_ids.append(rel['id'])

    return ent_ids[-1],rel_ids[-1]

def label_data(data, ent_id, rel_id):
    """
    标注数据
    """
    nlp = b_load_best_model('news')
    for sample in data:
        text = sample['text']
        doc = nlp(text)
        entities = []
        ent_dict = {}
        for ent in doc.ents:
            entities.append({
            'id': ent_id,
            'label': ent.label_,
            'start_offset': ent.start_char,
            'end_offset': ent.end_char
        })
            ent_dict[ent.start] = ent_id
            ent_id += 1
        relations = []
        for rel in doc._.rel.items():
            from_id,to_id = rel[0]
            from_id,to_id = ent_dict[from_id],ent_dict[to_id]
            for rel_type in rel[1].items():
                if rel_type[1] > 0.5:
                    relations.append({
                    'id': rel_id,
                    'from_id': from_id,
                    'to_id': to_id,
                    'type': rel_type[0],
                })
                    rel_id += 1
        sample['entities'] = entities
        sample['relations'] = relations

    b_save_dataset(data,'htmls_label.json')

client = get_doccano_client('local_doccano')
project_id = 3

b_doccano_project_export(client,project_id,'train.json')
b_doccano_project_export(client,project_id,'dev.json')

htmls = get_htmls()
data = get_htmls_content(htmls)

ent_id,rel_id = get_doc_id()

data = get_new_data(data)

label_data(data, ent_id, rel_id)

b_doccano_upload_project(client,project_id,'htmls_label.json',task='RelationExtraction')


