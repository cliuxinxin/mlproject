from hdfs.client import Client
import pandas as pd
from data_utils import *
from process_mysql_data_hdfs import *
class hdfsOperator(object):
    def __init__(self, client):
        self.client = client
    def read_hdfs_file(self, filename,origin_table):
        lines = []
        with self.client.read(filename, encoding='utf-8', delimiter='\n') as reader:
            for line in reader:
                lines.append(line.strip())
                if len(lines)==200:
                    deal_big_data(lines,origin_table)
                    lines=[]    
            deal_big_data(lines,origin_table)
    # 创建目录
    def mkdirs(self, hdfs_path):
        self.client.makedirs(hdfs_path)

    # 删除hdfs文件
    def delete_hdfs_file(self, hdfs_path):
        self.client.delete(hdfs_path)

    # 上传文件到hdfs
    def put_to_hdfs(self, local_path, hdfs_path):
        self.client.upload(hdfs_path, local_path, cleanup=True)

    # 从hdfs获取文件到本地
    def get_from_hdfs(self, hdfs_path, local_path):
        self.client.download(hdfs_path, local_path, overwrite=False)

    # 追加数据到hdfs文件
    def append_to_hdfs(self, hdfs_path, data):
        self.client.write(hdfs_path, data, overwrite=False, append=True, encoding='utf-8')

    # 覆盖数据写到hdfs文件
    def write_to_hdfs(self, hdfs_path, data):
        self.client.write(hdfs_path, data, overwrite=True, append=False, encoding='utf-8')

    # 移动或者修改文件
    def move_or_rename(self, hdfs_src_path, hdfs_dst_path):
        self.client.rename(hdfs_src_path, hdfs_dst_path)

    # 返回目录下的文件
    def list(self, hdfs_path):
        return self.client.list(hdfs_path, status=False)     
DFS = "http://bigdata03:7370/"

# 声明client
client = Client(DFS, root="/", timeout=10000, session=False)
# 声明hdfs
hdfs = hdfsOperator(client)
# 测试连接
try:

    hdfs.list('/')
except Exception as e:
    DFS = "http://bigdata01:7370/"
# 声明client
    client = Client(DFS, root="/", timeout=10000, session=False)
# 声明hdfs
    hdfs = hdfsOperator(client)




if __name__ == "__main__":
    pass
    # hdfs.read_hdfs_file()




        