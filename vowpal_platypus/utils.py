from internal import get_os, vw_hash_process_key

import re
import math
import os
import collections


def is_list(x):
    return isinstance(x, collections.Sequence) and not isinstance(x, basestring)

def mean(x):
    return sum(x) / float(len(x))

def clean(s):
      return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()

def safe_remove(f):
    os.system('rm -r ' + str(f) + ' 2> /dev/null')

def shuffle_file(filename, header=False):
    if get_os() == 'Mac':
        shuf = 'gshuf'
    else:
        shuf = 'shuf'
    if header:
        num_lines = sum(1 for line in open(filename))
        os.system('tail -n {} {} | {} > {}'.format(num_lines - 1, filename, shuf, filename + '_'))
    else:
        os.system('{} {} > {}'.format(shuf, filename, filename + '_'))
    return filename + '_'

def split_file(filename, num_cores, header=False):
    if num_cores > 1:
        print('Splitting {}...'.format(filename))
        num_lines = sum(1 for line in open(filename))
        if header:
            num_lines = sum(1 for line in open(filename))
            os.system('tail -n {} {} > {}'.format(num_lines - 1, filename, filename + '_'))
            filename = filename + '_'
        if get_os() == 'Mac':
            split = 'gsplit'
        else:
            split = 'split'
        os.system("{split} -d -l {lines} {filename} {filename}".format(split=split,
                                                                       lines=int(math.ceil(num_lines / float(num_cores))),
                                                                       filename=filename))
        return map(lambda x: filename + x,
                    map(lambda x: '0' + str(x) if x < 10 else str(x), range(num_cores)))
    else:
        os.system('cp {} {}00'.format(filename, filename))
        return [filename + '00']

def load_file(filename, process_fn, quiet=False):
    if not quiet:
        print 'Opening {}'.format(filename)
        num_lines = sum(1 for line in open(filename, 'r'))
        if num_lines == 0:
            raise ValueError('File is empty.')
        print 'Processing {} lines for {}'.format(num_lines, filename)
        i = 0
        curr_done = 0
    row_length = 0
    with open(filename, 'r') as filehandle:
        filehandle.readline()
        while True:
            item = filehandle.readline()
            if not item:
                break
            if not quiet:
                i += 1
                done = int(i / float(num_lines) * 100)
                if done - curr_done > 1:
                    print '{}: done {}%'.format(filename, done)
                    curr_done = done
            result = process_fn(item)
            if result is None:
                continue
            if row_length == 0:
                if is_list(result):
                    row_length = len(result)
                    data = {}
                else:
                    row_length = 1
                    data = []
            if row_length == 1:
                data.append(result)
            elif row_length == 2:
                key, value = result
                if data.get(key) is not None:
                    if not is_list(data[key]):
                        data[key] = [data[key]]
                    data[key].append(value)
                else:
                    data[key] = value
            elif row_length == 3:
                first_key, second_key, value = result
                if data.get(first_key) is None:
                    data[first_key] = {}
                if data[first_key].get(second_key) is not None:
                    if not is_list(data[first_key][second_key]):
                        data[first_key][second_key] = [data[first_key][second_key]]
                    data[first_key][second_key].append(value)
                else:
                    data[first_key][second_key] = value
            else:
                raise ValueError('I can only unpack files of length 3 or less and this was {}.'.format(row_length))
    return data

def vw_hash_to_vw_str(input_hash, logistic=False):
    vw_hash = input_hash.copy()
    vw_str = ''
    if vw_hash.get('label') is not None:
        label = vw_hash.pop('label')
        if logistic and (label == 0 or label == '0'):
            label = -1
        vw_str += str(label) + ' '
        if vw_hash.get('importance'):
            vw_str += str(vw_hash.pop('importance')) + ' '
    return vw_str + ' '.join(['|' + str(k) + ' ' + str(v) for (k, v) in zip(vw_hash.keys(), map(vw_hash_process_key, vw_hash.values()))])
