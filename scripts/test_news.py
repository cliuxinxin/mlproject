from data_utils import b_save_list_datasets
import hashlib
from newspaper import Article
from dragnet import extract_content_and_comments
from gne import GeneralNewsExtractor
import os
from bs4 import BeautifulSoup


newspaper_keywords = [
    'bohaishibei',
    'bbc.com',
    'zaobao.com',
    'ftchinese.com',
    'rfi.fr'
]

def mothod_choose(name):
    for keyword in newspaper_keywords:
        if keyword in name:
            return 'newspaper'
    return 'dragnet'

def download_html(url):
    article = Article(url)
    article.download()
    return article

def newspaper_process(article):
    article.parse()
    return article.text

def dragnet_process(article):
    return extract_content_and_comments(article.html)

def gne_process(article):
    extractor = GeneralNewsExtractor()
    result = extractor.extract(article.html)
    return result['title'] + '\n' +  result['content']

def get_article_from_url(url):
    method = mothod_choose(url)
    article = download_html(url)
    return article,method

def get_article_from_file(file):
    article = Article('')
    article.download(input_html=open(file, 'r').read())
    method = mothod_choose(file)
    return article,method

def parse_html(article,method):
    if method == 'newspaper':
        content = newspaper_process(article)
    else:
        content = dragnet_process(article)
    return content

def parse_date(bs,file):
    if 'cnBeta.COM' in file:
        return bs.find('div', class_='meta').find_all('span')[0].text
    if 'FT中文网' in file:
        return bs.find('span', class_='story-time').text.replace('更新于','')

def parse_source(bs,file):
    if 'cnBeta.COM' in file:
        return bs.find('span', class_='source').text.replace('稿源：','')
    if 'FT中文网' in file:
        return bs.find('span', class_='story-author').text.replace('FT中文网专栏作家','').replace('为FT中文网撰稿','')

htmls_path = '../assets/htmls/'

files = os.listdir(htmls_path)
files = [ file for file in files if file.endswith('.html')]
data = []

for file in files:
    file_path = htmls_path + file
    article,method = get_article_from_file(file_path)
    content = file + '\n' + parse_html(article, method)
    md5 = hashlib.md5(content.encode('utf-8')).hexdigest()
    bs = BeautifulSoup(article.html, 'html.parser')
    date = parse_date(bs,file)
    source = parse_source(bs,file)
    data.append({
        'method': method,
        'md5': md5,
        'date': date,
        'source': source,
        'text': content
    })

b_save_list_datasets(data, '../assets/htmls.json')

# file_path = htmls_path +'中亚行风光_ 观察人士_习近平撒币换声望，却得到高涨的排华效应.html'
# article,method = get_article_from_file(file_path)
# bs = BeautifulSoup(article.html, 'html.parser')

import pandas as pd
from data_utils import *

path = '../assets/test.xlsx'

df = pd.read_excel(path)

df.columns =['title', 'source', 'picture', 'summery', 'tag', 'status', 'title_detail', 'pubdate', 'no_important','writer', 'content']

df.dropna(subset=['title'],inplace=True)

df['text'] = df['title'] + '\n' + df['content']

df.drop(['no_important','picture','content'],axis=1,inplace=True)

b_save_df_datasets(df,'../assets/test.json')

# 下载数据
b_doccano_export_project(43,'../assets/news.json')

data = b_read_dataset('news.json')

new_data =[]

for entry in data:
    if len(entry['entities']) > 0:
        new_data.append(entry)

new_data = b_convert_ner_rel(new_data)

b_save_list_datasets(new_data,'../assets/news_data.json')
