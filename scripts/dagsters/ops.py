import os
from dagster import job, op, get_dagster_logger
from gne import GeneralNewsExtractor

ASSETS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'assets')


@op
def get_htmls():
    """
    从数据库中获取htmls文件目录
    """
    html_path = os.path.join(ASSETS_PATH,'htmls')
    htmls = os.listdir(html_path)
    htmls = [os.path.join(html_path,html) for html in htmls if html != '.DS_Store']
    get_dagster_logger().info(f"get_htmls: {htmls}")
    return htmls

@op
def get_htmls_content(htmls):
    """
    从htmls文件中获取content放到data
    """
    extractor = GeneralNewsExtractor()
    data = []
    for html in htmls:
        with open(html,'r',encoding='utf-8') as f:
            text = f.read()
        try:
            result = extractor.extract(text)
            data.append({
                'text':result['title'] + '\n' + result['content'],
                'author':result['author'],
                'name':html,
                'date':result['publish_time']
        })
        except:
            pass
    get_dagster_logger().info(f"get_htmls_content: {data}")
    return data

@job
def get_htmls_job():
    htmls = get_htmls()
    data = get_htmls_content(htmls)