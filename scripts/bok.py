from data_utils import *

def preprocess(df):
    """
    基本处理
    """
    # 增加长度项目
    df['length'] = df.apply(lambda x: len(x['content']), axis=1)


def read_data(file_path):
    """
    读取assets目录下的文件
    """
    with open(ASSETS_PATH + file_path, 'r') as f:
        text = f.readlines()
    df = pd.DataFrame(text)
    df.columns = ['content']
    return df 


def clean_text(text):
    """
    清洗文本
    """
    text = [x.strip() for x in text]
    return text


def is_chapter(text):
    """
    判断是否为章节
    """
    # 第一章 XXXXX
    if re.match(r'^第\d+章', text):
        return True
    else:
        return False


def count_words(text, words):
    """
    统计每个专业名词的频数
    """
    word_count = {}
    for word in words:
        word_count[word] = text.count(word)
    return word_count


def get_context(text,word):
    """
    获取某个专业名词的频数
    """
    # 查看文本中是否有word
    num = 30
    if word in text:
        start = text.index(word) - num if text.index(word) - num > 0 else 0
        end = text.index(word) + num if text.index(word) + num < len(text) else len(text)
        context = text[start:end]
        return context
    return ''

def get_text(text,word):
    """
    如果有关键词，就取得文本
    """
    if word in text:
        return text
    return ''

df = read_data('book.txt')
df = df.apply(clean_text)
preprocess(df)

# 去掉长度为0的行
df = df[df['length'] > 0]

# 统计总字数
print(df['length'].sum())

# 查看length最长的content
print(df[df['length'] == df['length'].max()])

# 查看最长长度
print(df['length'].max())

# 找到is_chapter为True的行
df[df['content'].apply(is_chapter)]

# 专业名词
with open(ASSETS_PATH + 'words', 'r') as f:
    # 去掉 \n
    words = f.readlines()
words = [x.strip() for x in words]

# 统计每个专业名词的频数
df['count_words'] = df['content'].apply(lambda x: count_words(x, words))

# 汇总每个专业名词的频数,并且按照频数排序
df['count_words'].apply(lambda x: pd.Series(x)).sum().sort_values(ascending=False)

count_words = df['count_words'].apply(lambda x: pd.Series(x)).sum().sort_values(ascending=False) 

# 得到 count_words 标题
for word in count_words.index.to_list():
    print(word)
    print('==' * 18)
    df['context'] = df['content'].apply(lambda x: get_text(x, word))
    for i in df[df['context'] != '']['context']:
        print(i)
    print('==' * 18)