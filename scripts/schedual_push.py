from apscheduler.schedulers.blocking import BlockingScheduler    # 引入模块


def task():
    '''定时任务'''
    print("hello world")

if __name__ == '__main__':
    scheduler = BlockingScheduler()

    # 添加任务
    scheduler.add_job(task, 'interval', second=3)
    scheduler.start()