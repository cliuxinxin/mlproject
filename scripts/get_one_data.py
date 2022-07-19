# 数据库抽取批量或一条数据然后进行纠错
import argparse
from matplotlib.pyplot import table
import pandas as pd
from data_utils import*
from mysql_utils import mysql_select_df


 
def get_parser():
        """
    脚本文件参数解析
    """
        parser = argparse.ArgumentParser(description="Read data from mysql and save to json")
        parser.add_argument('--task', default='', help='task name')
        parser.add_argument('--id', default='', help='id')
        parser.add_argument('--table', default='', help='save 100 records to a file')
        return parser

if __name__ == '__main__':  
  parser = get_parser()
  args = parser.parse_args()
  task = args.task
  id = args.id
  table = args.table
#   table = input("请输入数据库表名：") 
#   tablex=table.replace('final','test')
#   id = input("请输入id:")
#   task=table_config[tablex]["task"]
  tablex=table.replace('final','test')
  col = 'detail_content'
  source = 'source_website_address'
  sql = f"select * from {table} where id = '{id}'"
  # sql = f"select * FROM final_other_tender_bid_result WHERE CHAR_LENGTH(labels)>500  LIMIT 1"
  # sql = f"select {col},{source} FROM final_other_tender_bid_result WHERE CHAR_LENGTH(labels)>500  LIMIT 1"
  df = mysql_select_df(sql)
  df.to_json(DATA_PATH + f"{task}#{tablex}#{id}.json")
  data=df[[col,source]]
  a=p_process_df(data,task)
  b_save_df_datasets(a,"one_data.json")
  # 加入标签
  b_label_dataset_mult(task,'one_data.json',20)
# # 上传这条json到doccano
#   b_doccano_upload_by_task('one_data_label.json',task,'train')
#   b_doccano_upload('one_data_label.json',5)

