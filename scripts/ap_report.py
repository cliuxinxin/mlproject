import streamlit as st
from data_utils import *
from mysql_utils import *

def get_task_from_table(table):
    """
    从表名获取任务名
    """
    task = 'bid' if 'result' in table_name else 'tender'
    return task

'# 数据问题提交'

# 选择表名
table_name = st.selectbox('选择表名', [ 'final_other_tender_bid', 'final_procurement_bid','final_tender_bid','final_other_tender_bid_result','final_procurement_bid_result','final_tender_bid_result'])

# 输入id
id = st.text_input('输入id', '')

# 提取数据按钮
if st.button('提取数据'):
    task = get_task_from_table(table_name)
    col = 'detail_content'
    source = 'source_website_address'
    sql = f"select * from {table_name} where id = '{id}'"
    df = mysql_select_df(sql)
    if len(df) == 0:
        st.write('数据提取失败')
    else:
        tablex = table_name.replace('final', 'test')
        df.to_json(DATA_PATH + f"{task}#{tablex}#{id}.json")
        data=df[[col,source]]
        a=p_process_df(data,task)
        # 时间戳
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        file_name = f"one_data_{timestamp}"
        b_save_df_datasets(a,f"{file_name}.json")
        b_label_dataset_mult(task,f'{file_name}.json',20)
        # 读取 file_name_lable.json文件
        with open(ASSETS_PATH + f"{file_name}_label.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 显示数据
        st.write(data['label'])
        st.write('数据提取成功')


