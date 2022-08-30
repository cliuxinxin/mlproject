import re
import requests
from collections import Counter
from bs4 import BeautifulSoup


def get_html(url):
    """ 获取html """
    # obj = requests.get(url)
    # return obj.text

    try:
        obj = requests.get(url)
        code = obj.status_code
        if code == 200:
            # 防止中文正文乱码
            html = obj.content
            html_doc = str(html, 'utf-8')
            return html_doc
        return None
    except:
        return None





if __name__ == '__main__':
    url = 'https://cn.nytimes.com/asia-pacific/20220826/korea-abuse-brothers-home/'
    text = extract(url)
    print(text)