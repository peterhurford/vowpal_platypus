# Vowpal Platypus

**Vowpal Platypus** enables very quick, highly accurate, multicore, out-of-core, online learning in Python. VP is a general use, lightweight, Python Wrapper built on [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit/).


## Install

1. Install [vowpal_wabbit](https://github.com/JohnLangford/vowpal_wabbit/). Clone and run ``make``
2. Clone [vowpal_platypus](https://github.com/peterhurford/vowpal_platypus) and run `sudo python setup.py install`.

_(See [full instructions](https://github.com/peterhurford/vowpal_platypus/wiki/Installation) if necessary.)_

## Example

Predict survivorship on the Titanic [using the Kaggle data](https://www.kaggle.com/c/titanic):

```Python
from vowpal_platypus import linear_regression
from sklearn import metrics
import re
import os
import numpy

vw_model = linear_regression(name='Titanic', # Gives a name to the model file.
                             passes=40,      # How many online passes to do.
                             quadratic='ff', # Generates automatic quadratic features.
                             l1=0.0000001,   # L1 Regularization
                             l2=0.0000001)   # L2 Regularization

def clean(s):
  return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()

# VW trains on a file line by line. We need to define a function to turn each CSV line
# into an output that VW can understand.
def process_line(item):
    item = item.split(',')  # CSV is comma separated, so we unseparate it.
    features = [            # A set of features for VW to operate on.
                 'passenger_class_' + clean(item[2]),  # VP accepts individual strings as features.
                 'last_name_' + clean(item[3]),
                 {'gender': 0 if item[5] == 'male' else 1},  # Or VP can take a dict with a number.
                 {'siblings_onboard': int(item[7])},
                 {'family_members_onboard': int(item[8])},
                 {'fare': float(item[10])},
                 'embarked_' + clean(item[12])
               ]
    title = item[4].split(' ')
    if len(title):
        features.append('title_' + title[1])
    age = item[6]
    if age.isdigit():
        features.append({'age': int(item[6])})
    return {    # VW needs to process a dict with a label and then any number of feature sets.
        'label': 1 if item[1] == '1' else -1,
        'f': features   # The name 'f' for our feature set is arbitrary, but is the same as the 'ff' above that creates quadratic features.
    }

with vw_model.training():   # Start the online learning on each line of the file.
    with open('titanic_train.dat', 'r') as filehandle:
        filehandle.readline() # Throwaway header because it a CSV.
        while True:  # Iterate over the file.
            item = filehandle.readline()
            if not item:
                break
            vw_model.push_instance(process_line(item))  # Train the model on the processed line.
with vw_model.predicting():  # Start predicting.
    actuals = []
    with open('titanic_test.dat', 'r') as filehandle:
        filehandle.readline() # Throwaway header
        while True:
            item = filehandle.readline()
            if not item:
                break
            item = process_line(item)
            actuals.append(item['label'])  # Keep track of the labels to grade accuracy.
            vw_model.push_instance(item)   # Predict on the processed line.
all_results = zip(vw_model.read_predictions(), actuals)  # Collect the predictions made.
preds = map(lambda x: x[0], all_results)
actuals = map(lambda x: x[1], all_results)

d_preds = map(lambda x: -1 if x < 0.0 else 1, preds)
roc = str(metrics.roc_auc_score(numpy.array(d_preds), numpy.array(actuals)))
```

This produces a Titanic survival model with an AUC of 0.8471 in 0.52sec. That score is enough to get into the Top 100 on the leaderboard.


## Multicore Capabilities

Documentation coming soon.


## Deployment

Documentation coming soon.


## Available Models

**Nice interfaces:** `linear_regression`, `logistic_regression`, `als` (Alternating Least Squares Collaborative Filtering)
**Raw interface:** LDA, BFGS, Simple Nueral Nets, LRQ (nice interfaces coming soon)


## Credits, Contributions, and License

The base for this repository is [Vowpal Porpoise](https://github.com/josephreisinger/vowpal_porpoise) developed by Joseph Reisinger [@josephreisinger](http://twitter.com/josephreisinger), with further contributions by Austin Waters (austin.waters@gmail.com) and Daniel Duckworth (duckworthd@gmail.com).

This software is built using an [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0), the license used by the original contributors.

This repository also depends and is built upon [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit) developed by John Langford and other contributors, released under [a modified BSD License](https://github.com/JohnLangford/vowpal_wabbit/blob/master/LICENSE).
