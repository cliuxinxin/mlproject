import re
import datetime

def d_general_process(value,max_len):
    """
    一般符号处理
    """
    value = value.strip()
    value = value.replace('\n','')
    value = value.replace('\r','')
    value = value.replace('\t','')
    value = value.replace(' ','')
    return value[:max_len]

def d_amount_process(value):
    """
    一般金额处理
    如果是数字，就取第一个，然后有万的就乘以10000，有%的就去掉
    """
    value = value.replace(',','')
    numbers = re.findall(r'\d+\.\d+|\d+',value)
    if numbers:
        number = numbers[0]
        ten_thousand = re.findall(r'万',value)
        if ten_thousand:
            number = float(number) * 10000
            # 保留2位小数
            number = round(number,2)
            return str(number)
        persent = re.findall(r'%',value)
        if persent:
            number = number + '%'
            return ''
        return number
    return ''

def d_date_clean(value):
    """
    一般日期处理
    """
    value = re.findall(r'\d+', value)
    if value:
        # 如果第一位不是年
        if len(value[0]) != 4:
            year = datetime.datetime.now().strftime('%Y')
            # 如果年是两位数
            if len(value[0]) == 2 and int(value[0]) > 12:
                value[0] = '20' + value[0]
            else:
                # 第一位是月份
                value.insert(0,year)
        if len(value) < 6:
                for i in range(6 - len(value)):
                    value.append(0)
        # 全部转换成数字
        value = [int(i) for i in value]
        # 如果24点
        if value[3] == 24:
            value[3] = 23
            value[4] = 59
            value[5] = 59
        try:
            value = datetime.datetime(value[0], value[1], value[2], value[3], value[4], value[5])
            # 转换成标准时间格式
            value = value.strftime('%Y-%m-%d %H:%M:%S')
        except:
            value = ''
        return value
    


def clean_manager(task,col,value):
    func_name = 'clean_' + task + '_' + col
    max_len = 300
    value = d_general_process(value,max_len)
    if func_name in globals():
        return globals()[func_name](value)
    return value

def clean_bid_amount(value):
    """
    清洗出中标金额
    """
    value = d_amount_process(value)
    return value
    


def clean_tender_budget(value):
    """
    清洗出招标预算
    """
    value = d_amount_process(value) 
    return value

def clean_tender_sign_up_stime(value):
    """
    清洗出招标开始时间
    """
    value = d_date_clean(value)
    return value

def clean_tender_sign_up_etime(value):
    """
    清洗出招标结束时间
    """
    value = d_date_clean(value)
    return value

def clean_tender_tender_document_stime(value):
    """
    清洗出招标文件领取开始时间
    """
    value = d_date_clean(value)
    return value

def clean_tender_tender_document_etime(value):
    """
    清洗出招标文件领取结束时间
    """
    value = d_date_clean(value)
    return value

def clean_tender_bid_opening_time(value):
    """
    清洗出开标时间
    """
    value = d_date_clean(value)
    return value

def clean_tender_tender_etime(value):
    """
    清洗出投标截止时间
    """
    value = d_date_clean(value)
    return value
