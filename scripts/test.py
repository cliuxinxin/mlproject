from data_utils import * 

task = 'bid'

b_doccano_update_train_dev(task)

train_dev  = b_read_dataset('train_dev.json')

train_dev_label = b_read_dataset('train_dev_label.json')

all = train_dev + train_dev_label

data = []

for sample in all:
    text = sample['data']
    labels = sample['label']
    for label in labels:
        if label[2] == '中标金额':
            data.append(text[label[0]:label[1]])

data = set(data)

b_save_list_file(data,'data.txt')

def process(text):
    text = text.replace('\n','')
    text = text.replace('\r','')
    text = text.replace('\t','')
    text = text.replace(' ','')
    text = text.replace(',','')
    numbers = re.findall(r'\d+\.\d+|\d+',text)
    if numbers:
        number = numbers[0]
        ten_thousand = re.findall(r'万',text)
        if ten_thousand:
            number = float(number) * 10000
            # 保留2位小数
            number = round(number,2)
            return str(number)
        persent = re.findall(r'%',text)
        if persent:
            number = number + '%'
            return ''
        return number
    return ''

new_data = []

for sample in data:
    entry = {}
    entry['orig'] = sample
    entry['proc'] = process(sample)
    new_data.append(entry)

b_save_list_datasets(new_data,'new_data.json')

