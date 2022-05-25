from data_utils import * 
import pytest

def test_compare_have_tender():
    compare = b_read_db_compare()
    # 确保compare 的 task 中 有tender
    assert compare.groupby('task').count()['tender'] > 0
