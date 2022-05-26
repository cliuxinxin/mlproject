from data_utils import b_read_db_compare
import pytest
import os


def test_compare_have_tender():
    print(os.getcwd())
    compare = b_read_db_compare()
    assert compare.have_tender() is True