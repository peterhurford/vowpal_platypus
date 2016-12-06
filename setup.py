import setuptools
import setuptools.extension
from setuptools import setup, find_packages

setup(
    name = 'vowpal_platypus',
    version = '0.5',
    author = 'Peter Hurford',
    author_email = 'peter@peterhurford.com',
    description='Lightweight vowpal wabbit wrapper',
    license = 'Apache 2.0',
    keywords = 'machine learning regression vowpal_wabbit',
    url = 'https://github.com/peterhurford/vowpal_platypus',
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
