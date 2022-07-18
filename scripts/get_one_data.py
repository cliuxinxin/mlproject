# 数据库抽取批量或一条数据然后进行纠错
import argparse
import pandas as pd
from data_utils import*
from mysql_utils import mysql_select_df
table_config= {"test_other_tender_bid_result":{
        "target": "final_other_tender_bid_result",
        "task": "bid"
    },
    "test_other_tender_bid": {
        "target": "final_other_tender_bid", 
        "task": "tender"
    },
    "test_procurement_bid": {
        "target": "final_procurement_bid", 
        "task": "tender"
    },
    "test_procurement_bid_result": {
        "target": "final_procurement_bid_result", 
        "task": "bid"
    },
    "test_tender_bid": {
        "target": "final_tender_bid", 
        "task": "tender"
    },
    "test_tender_bid_result": {
        "target": "final_tender_bid_result", 
        "task": "bid"
    }}

if __name__ == '__main__':  
 
 
  table = input("请输入数据库表名：") 
  tablex=table.replace('final','test')
  id = input("请输入id:")
  task=table_config[tablex]["task"]
  col = 'detail_content'
  source = 'source_website_address'
  sql = f"select * from {table} where id = '{id}'"
  # sql = f"select * FROM final_other_tender_bid_result WHERE CHAR_LENGTH(labels)>500  LIMIT 1"
  # sql = f"select {col},{source} FROM final_other_tender_bid_result WHERE CHAR_LENGTH(labels)>500  LIMIT 1"
  df = mysql_select_df(sql)
  # 判断拿出的值是否为空
  # if len(df) >= 1:
  df.to_json(DATA_PATH + f"{task}#{tablex}#{id}.json")
  #   print("Get %s data from mysql and save to json" % len(df))
  # else:
  #   print("No data")
  data=df[[col,source]]

  a=p_process_df(data,task)
  b_save_df_datasets(a,"one_data.json")
  # 加入标签
  b_label_dataset_mult(task,'one_data.json',20)
# # 上传这条json到doccano
#   b_doccano_upload_by_task('one_data_label.json',task,'train')
#   b_doccano_upload('one_data_label.json',5)

