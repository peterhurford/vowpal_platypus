def log_loss(results):
     predicted = [min([max([x, 1e-15]), 1-1e-15]) for x in map(lambda x: float(x['predicted']), results)]
     target = [min([max([x, 1e-15]), 1-1e-15]) for x in map(lambda x: float(x['actual']), results)]
     return -(1.0 / len(target)) * sum([target[i] * log(predicted[i]) + (1.0 - target[i]) * log(1.0 - predicted[i]) for i in xrange(len(target))])

def rmse(results):
    return (sum(map(lambda x: (x['actual'] - x['predicted']) ** 2, results)) / len(results)) ** 0.5

def percent_correct(results):
    return sum(map(lambda x: x['actual'] == (-1 if x['predicted'] < 0 else 1), results)) / float(len(results)) * 100


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
