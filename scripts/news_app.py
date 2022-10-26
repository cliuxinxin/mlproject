import streamlit as st
from refac import b_load_best_model
import pandas as pd

# make the factory work
from rel_pipe import make_relation_extractor
# make the config work
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors

def process_entry(entry):
    ents  = {}
    text = entry['text']

    for ent in entry['entities']:
        ents[ent['id']] = text[ent['start_offset']:ent['end_offset']]

    events = []
    concepts = []
    views = []

    def find_ent_events(events,ent):
        for event in events:
            if event['ent'] == ent:
                return event
        return {}

    for relation in entry['relations']:
        relation_type = relation['type']
        if relation_type == '概念解释':
            concepts.append({
            'concept':ents[relation['from_id']],
            'explation':ents[relation['to_id']]
        })
            continue
        if relation_type == '实体观点':
            views.append({
            'ent':ents[relation['from_id']],
            'view':ents[relation['to_id']]
        })
            continue
        else:
            ent = relation['from_id']
            event = find_ent_events(events,ent)
            if event == {}: 
                events.append(event)
            event['ent'] = ents[relation['from_id']]
            if relation_type == '实体事件':
                event['event'] = ents[relation['to_id']]
            if relation_type == '实体时间':
                event['time'] = ents[relation['to_id']]
            if relation_type == '实体地点':
                event['place'] = ents[relation['to_id']]
     
    return events,concepts,views

def process_doc(doc):
    ent_id = 0
    rel_id = 0
    sample = {}
    sample['text'] = doc.text
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
    return sample


nlp = b_load_best_model('news')

st.title('信息提取')

text = st.text_area("输入信息", "",800)

if st.button("解析"):
    doc = nlp(text)
    entry = process_doc(doc)
    events,concepts,views = process_entry(entry)

    df_events = pd.DataFrame(events)
    df_concepts = pd.DataFrame(concepts)
    df_views = pd.DataFrame(views)


    "事件"
    df_events
    "概念"
    df_concepts
    "观点"
    df_views


