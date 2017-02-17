import pytest
from vowpal_platypus.utils import mean, clean, vw_hash_to_vw_str

def test_mean():
    assert mean([1, 2, 3]) == sum([1, 2, 3]) / 3.0
    assert mean([1]) == 1.0
    assert mean([-1, 1]) == 0.0

def test_clean():
    assert clean('hi') == 'hi'
    assert clean('') == ''
    assert clean('Hi') == 'hi'  # It lowercases
    assert clean('hi, how are you?') == 'hi how are you'  # It removes punctuation

def test_vw_hash_to_vw_str():
    {label: 0, f: ['feature'] }
    {label: 0.0, feature: ['feature'] }
    {label: 0, f: ['feature', 'other_feature'] }
    {label: 1, importance: 100, a: ['a', 'b', 'c'] }
    assert False # TODO
