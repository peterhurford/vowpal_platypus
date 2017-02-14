import pytest
from vowpal_platypus.utils import is_list, mean, clean, vw_hash_to_vw_str

def test_is_list():
    assert is_list([])
    assert is_list([1, 2, 3])
    assert is_list([1, 'a', 5])
    assert is_list([[]])
    assert not is_list({})
    assert not is_list('')
    assert not is_list('1a5')

def test_mean():
    assert mean([1, 2, 3]) == sum([1, 2, 3]) / 3.0
    assert mean([1]) == 1.0
    assert mean([-1, 1]) == 0.0

def test_clean():
    pass # TODO

def test_vw_hash_to_vw_str():
    pass # TODO
