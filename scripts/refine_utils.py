import requests

# refine url
url = '127.0.0.1:3333'

# 创建project
def create_project(project_name):
    url = 'http://' + url + '/command/core/create-project-from-upload'
    data = {
        'project_name': project_name,
        'project_file': '',
        'format':'text/json'
        }
    r = requests.post(url, data=data)
    return r.json()


