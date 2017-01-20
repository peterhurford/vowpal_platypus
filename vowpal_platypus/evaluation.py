from math import sqrt, log

def log_loss(results):
     predicted = [min([max([x, 1e-15]), 1-1e-15]) for x in map(lambda x: float(x[0]), results)]
     target = [min([max([x, 1e-15]), 1-1e-15]) for x in map(lambda x: float(x[1]), results)]
     return -(1.0 / len(target)) * sum([target[i] * log(predicted[i]) + (1.0 - target[i]) * log(1.0 - predicted[i]) for i in xrange(len(target))])

def rmse(results):
    return (sum(map(lambda x: (x[1] - x[0]) ** 2, results)) / float(len(results))) ** 0.5

def percent_correct(results):
    return sum(map(lambda x: x[1] == (0 if x[0] < 0.5 else 1), results)) / float(len(results)) * 100

def precision(results):
    true_positives = sum(map(lambda x: x[0] >= 0.5, filter(lambda x: x[1] == 1, results)))
    true_negatives = sum(map(lambda x: x[0] < 0.5, filter(lambda x: x[1] == 0, results)))
    false_positives = sum(map(lambda x: x[0] >= 0.5, filter(lambda x: x[1] == 0, results)))
    false_negatives = sum(map(lambda x: x[0] < 0.5, filter(lambda x: x[1] == 1, results)))
    return true_positives / max(float((true_positives + false_positives)), 1.0)

def recall(results):
    true_positives = sum(map(lambda x: x[0] >= 0.5, filter(lambda x: x[1] == 1, results)))
    true_negatives = sum(map(lambda x: x[0] < 0.5, filter(lambda x: x[1] == 0, results)))
    false_positives = sum(map(lambda x: x[0] >= 0.5, filter(lambda x: x[1] == 0, results)))
    false_negatives = sum(map(lambda x: x[0] < 0.5, filter(lambda x: x[1] == 1, results)))
    return true_positives / max(float((true_positives + false_negatives)), 1.0)

def f_score(results):
    precision_value = precision(results)
    recall_value = recall(results)
    return 2 * ((precision_value * recall_value) / max(precision_value + recall_value, 0.000001))

def mcc(results):
    true_positives = sum(map(lambda x: x[0] >= 0.5, filter(lambda x: x[1] == 1, results)))
    true_negatives = sum(map(lambda x: x[0] < 0.5, filter(lambda x: x[1] == 0, results)))
    false_positives = sum(map(lambda x: x[0] >= 0.5, filter(lambda x: x[1] == 0, results)))
    false_negatives = sum(map(lambda x: x[0] < 0.5, filter(lambda x: x[1] == 1, results)))
    return ((true_positives * true_negatives) - (false_positives * false_negatives)) / sqrt(float(max((true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_positives) * (true_negatives + false_negatives), 1.0)))

def average_accuracy(results):
    true_positives = sum(map(lambda x: x[0] >= 0.5, filter(lambda x: x[1] == 1, results)))
    true_negatives = sum(map(lambda x: x[0] < 0.5, filter(lambda x: x[1] == 0, results)))
    false_positives = sum(map(lambda x: x[0] >= 0.5, filter(lambda x: x[1] == 0, results)))
    false_negatives = sum(map(lambda x: x[0] < 0.5, filter(lambda x: x[1] == 1, results)))
    return 0.5 * ((true_positives / float(true_positives + false_negatives)) + (true_negatives / float(true_negatives + false_positives)))


def auc(results):
    def _tied_rank(x):
        sorted_x = sorted(zip(x,range(len(x))))
        r = [0 for k in x]
        cur_val = sorted_x[0][0]
        last_rank = 0
        for i in range(len(sorted_x)):
            if cur_val != sorted_x[i][0]:
                cur_val = sorted_x[i][0]
                for j in range(last_rank, i): 
                    r[sorted_x[j][1]] = float(last_rank+1+i)/2.0
                last_rank = i
            if i==len(sorted_x)-1:
                for j in range(last_rank, i+1): 
                    r[sorted_x[j][1]] = float(last_rank+i+2)/2.0
        return r

    def _auc(actual, posterior):
        r = _tied_rank(posterior)
        num_positive = len([0 for x in actual if x==1])
        num_negative = len(actual)-num_positive
        sum_positive = sum([r[i] for i in range(len(r)) if actual[i]==1])
        auc = ((sum_positive - num_positive*(num_positive+1)/2.0) /
               (num_negative*num_positive))
        return auc

    preds = map(lambda x: x[0], results)
    actuals = map(lambda x: x[1], results)
    return _auc(actuals, preds)
