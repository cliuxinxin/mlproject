from selenium import webdriver
import random

# 创建safari浏览器对象
browser = webdriver.Safari()

browser.get('https://www.cnbeta.com/articles/tech/1282989.htm')

path = '/Users/liuxinxin/Downloads/'

# 保存为html格式,随机生成名字
name = str(random.randint(0, 100000)) + '.html'
with open(path + name, 'w', encoding='utf-8') as f:
    f.write(browser.page_source)
