import pytest
import os
import os.path
from vowpal_platypus.utils import shuffle_file, safe_remove, split_file

def test_shuffle_file():
    with open('/tmp/testfile.dat', 'w') as filehandle:
        filehandle.write('\n'.join(map(str, range(10000))))
    shuffled_file = shuffle_file('/tmp/testfile.dat')
    contents = map(int, open(shuffled_file, 'r').readlines())
    os.remove('/tmp/testfile.dat')
    os.remove(shuffled_file)
    assert len(contents) == 10000
    assert contents != range(10000)

def test_safe_remove_file():
    open('/tmp/testfile.dat', 'w')
    assert os.path.exists('/tmp/testfile.dat')
    safe_remove('/tmp/testfile.dat')
    assert not os.path.exists('/tmp/testfile.dat')

def test_safe_remove_directory():
    os.makedirs('/tmp/testdir')
    assert os.path.exists('/tmp/testdir')
    safe_remove('/tmp/testdir')
    assert not os.path.exists('/tmp/testdir')

def test_safe_remove_glob():
    os.makedirs('/tmp/testdir')
    open('/tmp/testdir/testfile_a.dat', 'w')
    open('/tmp/testdir/testfile_b.dat', 'w')
    assert os.path.exists('/tmp/testdir/testfile_a.dat')
    assert os.path.exists('/tmp/testdir/testfile_b.dat')
    safe_remove('/tmp/testdir/*')
    assert not os.path.exists('/tmp/testdir/testfile_a.dat')
    assert not os.path.exists('/tmp/testdir/testfile_b.dat')
    assert os.path.exists('/tmp/testdir')
    safe_remove('/tmp/testdir')
    assert not os.path.exists('/tmp/testdir')

def test_split_file():
    with open('/tmp/testfile.dat', 'w') as filehandle:
        filehandle.write('\n'.join(map(str, range(10000))))
    split_files = split_file('/tmp/testfile.dat', num_cores=10)
    contents = [open(f, 'r').readlines() for f in split_files]
    assert len(contents) == 10
    assert len(contents[0]) == 1000
    assert map(int, contents[0][0:10]) == range(10)
    os.remove('/tmp/testfile.dat')
    [os.remove(f) for f in split_files]
