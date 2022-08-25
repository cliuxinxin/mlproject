from tqdm import tqdm
from data_utils import *
from mysql_utils import *

from dragnet import extract_content, extract_content_and_comments

    
path = ASSETS_PATH + 'htmls/'

files = os.listdir(path)

file = files[9]

with open(path + file,'r') as f:
    html = f.read()







sql = 'select * from final_procurement_bid limit 10'
df = mysql_select_df(sql)

df.columns
html = df.detail_content.values[0]

p_filter_tags(html)
extract_content_and_comments(html)





