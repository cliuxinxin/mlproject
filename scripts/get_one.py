from data_utils import p_process_df,DATA_PATH,b_load_best_model,b_doccano_upload_by_task,b_save_df_datasets,project_configs
from mysql_utils import mysql_select_df
import time
import argparse
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="Read data from mysql and save to json")
    parser.add_argument('--task', default='bid', help='task name')
    parser.add_argument('--id', default='none', help='id')
    return parser

def label_data(nlp,data):
    doc = nlp(data)
    return [[ent.start_char,ent.end_char,ent.label_] for ent in doc.ents]

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    task = args.task
    id = args.id
    task = 'bid'
    id = 'ah-anqing-003002009b59354df-0305-4164-b6e1-a4427cbd9716zbgs_002'
    table = project_configs[task]['table']
    col = project_configs[task]['col']
    source = project_configs[task]['source']
    sql = "select c%,source from %s where id = '%s'" % (table,id)
    df = mysql_select_df(sql)
    if len(df) >= 1:
        df.to_json(DATA_PATH + task + '_' + id + '.json')
        df = p_process_df(df,task)
        nlp = b_load_best_model(task)
        df['label'] = df['data'].apply(lambda x:label_data(nlp,x))
        b_save_df_datasets(df,'one_data.json')
        b_doccano_upload_by_task('one_data.json',task,'train')
        print("Get %s data from mysql and save to json" % len(df))
    else:
        print("No data")