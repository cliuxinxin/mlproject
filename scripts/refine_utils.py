from refac import *
from requests_toolbelt import MultipartEncoder
import requests
import json

class OpenRefine():
    def __init__(self, server_url):
        self.server_url = server_url
        self.project_url = ''
        self.err = None

    def get_token(self):
        t_url = self.server_url + '/command/core/get-csrf-token'
        res = requests.get(t_url)
        jdat = json.loads(res.text)
        token = jdat.get('token', '')
        return token

    def get_project_id(self,content):
        pid = ''
        pat = 'project\?project=(\d+)'
        mdat = re.findall (pat, content)
        if mdat:
            pid = mdat[0]
        return pid

    def upload(self,file_name,file_path, project_name='untitled'):     
        '''上传文件并生成项目
        fname: 文件名
        project_name: 项目名称，如果为空则以文件名自动命名
        '''
        # 生成提交地址
        u_url = self.server_url + '/command/core/create-project-from-upload'
        
        # 自动生成项目名

        try:
            self.project_url = ''
        
            # 生成数据
            print('正在组织数据...')
            format = 'text/line-based/*sv'
            f_format = 'text/plain'

            mdat = MultipartEncoder(
                fields={'project-file': (file_name, open(file_path, 'rb'), f_format), 
                        'project-name': project_name,
                        'format': format}
                )
            print('mdat.content_type:', mdat.content_type)
            print('mdat:', mdat)

            # 获取令牌
            token = self.get_token()
            # 添加令牌
            url = '%s?csrf_token=%s' % (u_url, token)

            print('正在上传...')
            ret = requests.post(url, 
                                data=mdat.to_string(),
                                headers={'Content-Type': mdat.content_type},
                                allow_redirects=False)

            print('上传结果:',ret)
            print('ret headers:', ret.headers)

            # 提取项目号及项目链接 改为直接从302的header提取
            purl = ret.headers.get('Location', '')
            self.project_url = purl
            project_id = self.get_project_id(purl)
        except Exception as e:
            self.err = e
            print(e)
            project_id = ''

        return project_id

    def upload_path(self,path):
        file_name = os.path.basename(path)
        project_name = os.path.splitext(file_name)[0]
        return self.upload(file_name, path, project_name)


# file_name = 'events.csv'
# openrefine = OpenRefine(server_url)
# project_id = openrefine.upload(file_name,ASSETS_PATH + file_name, project_name='events2')
