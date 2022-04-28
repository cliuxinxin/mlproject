from data_utils import * 

def redit_preocess_post(file):
    """
    处理post，讲title和selftext结合，并且保存为_imp.json文件方便导入    
    """

    file_name = file.split('.')[0]

    posts = b_read_dataset(file)

    for sample in posts:
        try:
            title = sample['title']
            selftext = sample['selftext']
        except:
            title = ''
            selftext = ''
        sample['text'] = title + '\n' + selftext
        sample['title'] = title
        sample['selftext'] = selftext
    # 去掉title和selftext
        del sample['title']
        del sample['selftext']

    b_save_list_datasets(posts,file_name + '_imp.json')

def redit_process_comments(file):
    """
    处理reddit的comments数据，将body改成text，并且保存为_imp.json文件方便导入 
    """
    file_name = file.split('.')[0]
    comments = b_read_dataset(file)

    for sample in comments:
        try:
            body = sample['body']
        except:
            body = ''
    # 把body改成text
        sample['text'] = body
        sample['body'] = body
    # 去掉body
        del sample['body']

    b_save_list_datasets(comments,file_name + '_imp.json')