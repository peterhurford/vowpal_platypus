import setuptools
import setuptools.extension
from setuptools import setup, find_packages

setup(
    name = 'vowpal_porpoise',
    version = '0.4',
    author = 'Joseph Reisinger',
    author_email = 'joeraii@gmail.com',
    description='Lightweight vowpal wabbit wrapper',
    license = 'BSD',
    keywords = 'machine learning regression vowpal_wabbit',
    url = 'https://github.com/josephreisinger/vowpal_porpoise',
    packages = find_packages(),
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    install_requires = [    # dependencies
    ],
    tests_require = [    # test dependencies
    ]
)
