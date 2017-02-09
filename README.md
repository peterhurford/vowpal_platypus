# Vowpal Platypus <a href="https://github.com/peterhurford/vowpal_platypus/blob/master/CHANGELOG.md"><img src="https://img.shields.io/github/tag/peterhurford/vowpal_platypus.svg"></a>

**Vowpal Platypus** is a general use, lightweight Python wrapper built on [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit/), that uses online learning to acheive great results.

VP is...

* **...quick,** generating [MovieLens predictions](https://github.com/peterhurford/vp_examples/blob/master/als/vp/runner.py) with just a few _nanoseconds_ per prediction on a 40 core EC2.
* **...accurate,** acheiving AUC > 0.9 with [a Titanic model](https://github.com/peterhurford/vp_examples/blob/master/titanic/vp/kaggle.py) that processes, trains, and predicts all in under a second on a laptop.
* **...versitile,** implementing logistic regression, linear regression, collaborative filtering (ALS), simple nueral nets, LDA, and other algorithms.
* **...lightweight,** with no dependencies other than Python, installing on a Macbook pro in 0.3 seconds.
* **...multicore,** scaling linearly across any number of cores, being used for hundreds of GB of data.
* **...out-of-core,** bottlenecked by CPU and IO rather than RAM.

**[See demo code here](https://github.com/peterhurford/vp_examples)** showing detailed implementations and benchmarks for MovieLens ALS, Criteo ad click prediction, NumerAI stock prediction, and Titanic survival.


## Install

1. Install [vowpal_wabbit](https://github.com/JohnLangford/vowpal_wabbit/). Clone and run ``make``
2. Clone [vowpal_platypus](https://github.com/peterhurford/vowpal_platypus) and run `sudo python setup.py install`. You will also need to install [Retrying](https://pypi.python.org/pypi/retrying), [Pathos](https://github.com/uqfoundation/pathos), and [Dill](https://github.com/uqfoundation/dill/).

_(See [full installation instructions](https://github.com/peterhurford/vowpal_platypus/wiki/Installation) if necessary.)_


## Demo

Predict survivorship on the Titanic [using the Kaggle data](https://www.kaggle.com/c/titanic):

```Python
from vowpal_platypus import run                        # The run function is the main function for running VP models.
from vowpal_platypus.models import logistic_regression # vowpal_platypus.models is where all the models are imported from.
from vowpal_platypus.evaluation import auc             # vowpal_platypus.evaluation can import a lot of evaluation functions, like AUC.
from vowpal_platypus.utils import clean                # vowpal_platypus.utils has some useful utility functions.

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
        features.append('title_' + title[1])  # Add a title feature if they have one.
    age = item[6]
    if age.isdigit():
        features.append({'age': int(item[6])})
    return {    # VW needs to process a dict with a label and then any number of feature sets.
        'label': int(item[1] == '1'),
        'f': features   # The name 'f' for our feature set is arbitrary, but is the same as the 'ff' above that creates quadratic features.
    }

# Train a logistic regression model on Titanic survival.
# The `run` function will automatically generate a train - test split.
run(logistic_regression(name='Titanic',    # Gives a name to the model file.
                        passes=3,          # How many online passes to do.
                        quadratic='ff',    # Generates automatic quadratic features.
                        nn=5),             # Add a neural network layer with 5 hidden units.
    'titanic/data/titanic.csv',     # File with the data (will automatically be split into random train and test)
    line_function=process_line,     # Function to process each line of the file
    evaluate_function=auc)          # Function to evaluate results
```

This produces a Titanic survival model with an AUC of 0.7241 (on the Kaggle holdout validation set) in 0.16sec.


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
