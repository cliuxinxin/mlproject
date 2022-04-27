import configparser
import json
import praw

def d_parse_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

project_configs = d_parse_config()

reddit = praw.Reddit(
    client_id=project_configs['reddit']['client_id'],
    client_secret=project_configs['reddit']['client_secret'],
    user_agent='<colab>'
)

sub_reddit = reddit.subreddit("CryptoCurrency")

# 下载所有的文章
def d_get_all_posts(sub_reddit):
    for post in sub_reddit.stream.submissions():
        yield post

# 下载文章的评论
def d_get_all_comments(post):
    for comment in post.comments:
        yield comment

# 将所有的文章转换成字典
def d_convert_post_to_dict(post):
    return {
        'title': post.title,
        'selftext': post.selftext,
        'url': post.url,
        'created_utc': post.created_utc,
        'score': post.score,
        'num_comments': post.num_comments,
        'id': post.id,
        'permalink': post.permalink,
        'url': post.url,
        'subreddit': post.subreddit.display_name
    }


# 将所有的评论转换成字典
def d_convert_comments_to_dict(comment):
    return {
        # 如果没有body，则返回None
        try:
            'body': comment.body,
        except:
            'body': None,
        'created_utc': comment.created_utc,
        'score': comment.score,
        'id': comment.id,
        'permalink': comment.permalink,
        'post_id': comment.link_id,
        'subreddit': comment.subreddit.display_name
    }

# 将字典转换成json
def d_convert_to_json(post):
    return json.dumps(post)
    
# 将所文章保存
def d_save_posts(posts):
    with open('../assets/posts.json', 'a') as f:
        for post in posts:
            # 保存所有的评论
            with open('../assets/comments.json', 'a') as f2:
                for comment in d_get_all_comments(post):
                    f2.write(d_convert_to_json(d_convert_comments_to_dict(comment)))
                    f2.write('\n')
            f.write(d_convert_to_json(d_convert_post_to_dict(post)))
            f.write('\n')

posts = d_get_all_posts(sub_reddit)
d_save_posts(posts)