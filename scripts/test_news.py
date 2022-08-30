from newspaper import Article
from dragnet import extract_content_and_comments
from gne import GeneralNewsExtractor


newspaper_keywords = [
    'bohaishibei',
    'bbc.com',
    'zaobao.com',
    'ftchinese.com',
    'rfi.fr'
]

def mothod_choose(url):
    for keyword in newspaper_keywords:
        if keyword in url:
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



url = 'http://www.dapenti.com/blog/more.asp?name=xilei&id=166509'

method = mothod_choose(url)
article = download_html(url)

if method == 'newspaper':
    print(newspaper_process(article))
else:
    print(dragnet_process(article))



