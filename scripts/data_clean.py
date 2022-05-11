import re

def d_general_process(text):
    text = text.strip()
    text = text.replace('\n','')
    text = text.replace('\r','')
    text = text.replace('\t','')
    text = text.replace(' ','')
    return text


def clean_manager(task,col,data):
    func_name = 'clean_' + task + '_' + col
    if func_name in globals():
        return globals()[func_name](data)
    return data

def clean_bid_amount(data):
    """
    清洗出中标金额，如果是数字，就取第一个，然后有万的就乘以10000，有%的就去掉
    """
    data = d_general_process(data)
    data = data.replace(',','')
    numbers = re.findall(r'\d+\.\d+|\d+',data)
    if numbers:
        number = numbers[0]
        ten_thousand = re.findall(r'万',data)
        if ten_thousand:
            number = float(number) * 10000
            # 保留2位小数
            number = round(number,2)
            return str(number)
        persent = re.findall(r'%',data)
        if persent:
            number = number + '%'
            return ''
        return number
    return ''

def clean_bid_project_name(data):
    """
    清洗出中标项目名称
    """
    data = d_general_process(data)
    return data

def clean_bid_notice_num(data):
    """
    清洗出中标公告号
    """
    data = d_general_process(data)
    return data