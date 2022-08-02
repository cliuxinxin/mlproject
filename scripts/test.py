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

keyword = '项目建议书'

sql = f'select * from {tabels[0]} where title like "{keyword}%" limit 1'

df = mysql_select_df(sql)

df.columns
    
    



