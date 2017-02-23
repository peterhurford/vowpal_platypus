import pytest
from vowpal_platypus.evaluation import log_loss

perfect_results = [(1, 1), (0, 0)]
no_signal_results = [(0.5, 1), (0.5, 0)]
good_results = [(0.1, 0), (0.9, 1), (0.8, 1), (0.2, 0)]

def equal_within_epsilon(x, y):
    return abs(x - y) < pow(10, -8)

def test_log_loss():
    assert equal_within_epsilon(log_loss(perfect_results), 0.0)
    assert equal_within_epsilon(log_loss(no_signal_results), 0.301029996)
    assert equal_within_epsilon(log_loss(good_results), 0.068636236)

# def test_rmse():
#     assert False # TODO

# def test_percent_correct():
#     assert False # TODO

# def test_true_positives():
#     assert False # TODO

# def test_true_negatives():
#     assert False # TODO

# def test_false_negatives():
#     assert False # TODO

# def test_false_positives():
#     assert False # TODO

# def test_tpr():
#     assert False # TODO

# def test_tnr():
#     assert False # TODO

# def test_fnr():
#     assert False # TODO

# def test_fpr():
#     assert False # TODO

# def test_precision():
#     assert False # TODO

# def test_f_score():
#     assert False # TODO

# def test_mcc():
#     assert False # TODO

# def test_average_accuracy():
#     assert False # TODO

# def test_auc():
#     assert False # TODO
