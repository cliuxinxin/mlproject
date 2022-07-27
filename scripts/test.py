from tqdm import tqdm
from data_utils import *
from mysql_utils import *
from func_timeout import func_set_timeout

conn = mysql_connect()


tabels = [
    'test_other_tender_bid',
    'test_procurement_bid',
    'test_tender_bid'
]

def read_file(file):
    df = pd.read_csv(ASSETS_PATH + file,header=None)
    df.columns = ['source_website_name','table']
    df['pk'] = df['source_website_name'] + '_' + df['table']
    return df

def write_data(df,file):
    file = ASSETS_PATH + file
    df.to_csv(file, mode='a', header=False, index=False)

df_source = read_file('table_source.csv')
df_finish = read_file('finish.csv')
df_source = df_source[~df_source['pk'].isin(df_finish['pk'])]


for source,table,pk in tqdm(df_source.values):
    sql = f"select id from {table} where source_website_name = '{source}' limit 10"
    df_id = mysql_select_df(sql)
    if len(df_id) == 1:
        sql = f"select '{table}' as table_name,id,source_website_name,source_website_address,detail_content from {table} where id = '{df_id.id.values[0]}'"
        df = mysql_select_df(sql)
        write_data(df, 'dev_add.csv')
    else:
        train = int(len(df_id) * 0.8)
        sql = f"select '{table}' as table_name,id,source_website_name,source_website_address,detail_content from {table} where id in {tuple(df_id.id.values.tolist())}"
        df = mysql_select_df(sql)
        write_data(df[:train], 'train_add.csv')
        write_data(df[train:], 'dev_add.csv') 
    finish_df = pd.DataFrame({'source_website_name':[source],'table':[table]})
    write_data(finish_df,'finish.csv')
    
    



