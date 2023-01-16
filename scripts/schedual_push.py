from redis_utils import redis_,diff_key,redis_push_diff,ori_tar_configs,tag_key,redis_push_tag,tag_configs,redis_push_all_tender
import time


def is_redis_not_enough(key,length):
    return redis_.llen(key) < length

while True:
    if is_redis_not_enough(diff_key,1000):
        try:
            # redis_push_diff(ori_tar_configs)
            redis_push_all_tender()
            print(f'There is {redis_.llen(diff_key)} data in redis')
        except:
            continue
    if is_redis_not_enough(tag_key,1000):
        try:
            redis_push_tag(tag_configs)
            print(f'There is {redis_.llen(tag_key)} data in redis')
        except:
            continue
    break
    time.sleep(60)