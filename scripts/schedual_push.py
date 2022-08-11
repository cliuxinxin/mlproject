from redis_utils import redis_,diff_key,redis_push_diff,ori_tar_configs
import time


def is_redis_not_enough(key,length):
    return redis_.llen(key) < length

while True:
    if is_redis_not_enough(diff_key,1000):
        try:
            redis_push_diff(ori_tar_configs)
            print(f'There is {redis_.llen(diff_key)} data in redis')
        except:
            continue
    time.sleep(60)