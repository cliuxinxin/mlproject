from refac import *

path = assets_path('news_test.json')

with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

nlp = b_load_best_model('news')

events = []
concepts = []
views = []

def find_ent_events(events,ent):
        for event in events:
            if event['ent_id'] == ent:
                return event
        return {}


def save_data(from_ent_id,to_ent_id,type_label,entry):
    if type_label == '概念解释':
            concepts.append({
                'concept':ents[from_ent_id],
                'explation':ents[to_ent_id],
                'class':entry['class'],
                'source':entry['name']

            })
            return
    if type_label == '实体观点':
            views.append({
                'ent':ents[from_ent_id],
                'view':ents[to_ent_id],
                'class':entry['class'],
                'source':entry['name']
            })
            return
    else:
        ent = from_ent_id
        event = find_ent_events(one_events,ent)
        if event == {}:
            one_events.append(event)
            event['ent_id'] = ent
            event['ent'] = ents[ent]
            event['class'] = entry['class']
            event['source'] = entry['name']
        if type_label == '实体事件':
            event['event'] = ents[to_ent_id]
        if type_label == '实体时间':
            event['time'] = ents[to_ent_id]
        if type_label == '实体地点':
            event['place'] = ents[to_ent_id]
        

for entry in data:
    doc = nlp(entry['content'])

    one_events = []

    ents = {}
    types = {}

    for ent in doc.ents:
        ents[ent.start] = ent.text
        types[ent.start] = ent.label_

    for key,val in doc._.rel.items():
        from_ent_id,to_ent_id = key
        for type_label,type_value in val.items():
            if type_value > 0.5:
                save_data(from_ent_id,to_ent_id,type_label,entry)
    
    events.extend(one_events)

df_events = pd.DataFrame(events)
df_concepts = pd.DataFrame(concepts)
df_views = pd.DataFrame(views)

df_events.drop(columns=['ent_id'],inplace=True)

df_events.to_csv(assets_path('events.csv'),index=False)
df_concepts.to_csv(assets_path('concepts.csv'),index=False)
df_views.to_csv(assets_path('views.csv'),index=False)




