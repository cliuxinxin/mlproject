from data_utils import *
from mysql_utils import *
from data_clean_new import clean_manager

import pandas as pd

data = range(6)

df = pd.DataFrame(data)

df = df + 1

df.loc[1:,0] = 0