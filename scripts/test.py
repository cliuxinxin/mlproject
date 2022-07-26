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

@func_set_timeout(180)
def get_source_data(table, s):
    sql = f"select '{table}' as table_name,id,source_website_name,source_website_address,detail_content from {table} where source_website_name = '{s}' limit 3"
    df = pd.read_sql(sql,con=conn)
    return df

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
df_wrong = read_file('wrong_table.csv')
df_source = df_source[~df_source['pk'].isin(df_finish['pk'])]
df_source = df_source[~df_source['pk'].isin(df_wrong['pk'])]


for source,table,pk in tqdm(df_source.values):
    try:
        df = get_source_data(table, source)
    except:
        wrong_df = pd.DataFrame({'source_website_name':[source],'table':[table]})
        write_data(wrong_df,'wrong_table.csv')
        continue
    write_data(df[:2], 'train_add.csv')
    write_data(df[2:], 'dev_add.csv')
    finish_df = pd.DataFrame({'source_website_name':[source],'table':[table]})
    write_data(finish_df,'finish.csv')
    
    
        


    



