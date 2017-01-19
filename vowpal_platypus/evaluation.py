def log_loss(results):
     predicted = [min([max([x, 1e-15]), 1-1e-15]) for x in map(lambda x: float(x['predicted']), results)]
     target = [min([max([x, 1e-15]), 1-1e-15]) for x in map(lambda x: float(x['actual']), results)]
     return -(1.0 / len(target)) * sum([target[i] * log(predicted[i]) + (1.0 - target[i]) * log(1.0 - predicted[i]) for i in xrange(len(target))])

def rmse(results):
    return (sum(map(lambda x: (x['actual'] - x['predicted']) ** 2, results)) / len(results)) ** 0.5

def percent_correct(results):
    return sum(map(lambda x: x['actual'] == (-1 if x['predicted'] < 0 else 1), results)) / float(len(results)) * 100
