from data_utils import b_doccano_bak_train_dev

tasks = ['tender','bid']

for task in tasks:
    print("task:",task)
    b_doccano_bak_train_dev(task)

