from internal import get_os, vw_hash_process_key, to_str

import re
import math
import os
import collections

def mean(x):
    return sum(x) / float(len(x))


def clean(s):
      """Return lowercased input with no punctuation."""
      return ' '.join(re.findall(r"\w+", s, flags = re.UNICODE | re.LOCALE)).lower()


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


# TODO: DRY?
def load_cassandra_query(query, cassandra_session, process_fn, quiet=False, header=True):
    row_length = 0
    data = None  # Initialize `data` so that it can be returned if there are no results.
    for row in cassandra_session.execute(query):
        result = process_fn(row)
        if row_length == 0:
            if isinstance(result, list):
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
                if not isinstance(data[key], list):
                    data[key] = [data[key]]
                data[key].append(value)
            else:
                data[key] = value
        elif row_length == 3:
            first_key, second_key, value = result
            if data.get(first_key) is None:
                data[first_key] = {}
            if data[first_key].get(second_key) is not None:
                if not isinstance(data[first_key][second_key], list):
                    data[first_key][second_key] = [data[first_key][second_key]]
                data[first_key][second_key].append(value)
            else:
                data[first_key][second_key] = value
        else:
            raise ValueError('I can only unpack files of length 3 or less and this was {}.'.format(row_length))
    return data


def load_file(filename, process_fn, quiet=False, header=True):
    if not quiet:
        print 'Opening {}'.format(filename)
        num_lines = sum(1 for line in open(filename, 'r'))
        if num_lines == 0:
            raise ValueError('File is empty.')
        print 'Processing {} lines for {}'.format(num_lines, filename)
        i = 0
        curr_done = 0
    row_length = 0
    data = None  # Initialize `data` so that it can be returned if there are no results.
    with open(filename, 'r') as filehandle:
        if header is True:
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
                if isinstance(result, list):
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
                    if not isinstance(data[key], list):
                        data[key] = [data[key]]
                    data[key].append(value)
                else:
                    data[key] = value
            elif row_length == 3:
                first_key, second_key, value = result
                if data.get(first_key) is None:
                    data[first_key] = {}
                if data[first_key].get(second_key) is not None:
                    if not isinstance(data[first_key][second_key], list):
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
        if not (isinstance(label, int) or isinstance(label, float) or (isinstance(label, basestring) and label.isdigit())):
            raise ValueError('Labels passed to VP must be numeric.')
        if logistic and (label == 0 or label == '0'):
            label = -1
        vw_str += to_str(label) + ' '
        if vw_hash.get('importance'):
            vw_str += to_str(vw_hash.pop('importance')) + ' '
    if not all(map(lambda x: isinstance(x, basestring) and len(x) == 1, vw_hash.keys())):
        raise ValueError('Namespaces passed to VP must be length-1 strings.')
    return vw_str + ' '.join(['|' + to_str(k) + ' ' + to_str(v) for (k, v) in zip(vw_hash.keys(), map(vw_hash_process_key, vw_hash.values()))])


def split_object(obj, num_parts):
    leng = len(obj)
    if leng < num_parts:
        raise ValueError('Object passed to `split_object` is smaller (length {}) than the number of splits ({})'.format(leng, num_parts))
    if num_parts == 1:
        return [obj]
    if isinstance(obj, list):
        return split_list(obj, num_parts)
    elif isinstance(obj, dict):
        return split_dict(obj, num_parts)
    else:
        raise ValueError('Object passed to `split_object` should be a list or a dictionary. Instead a {} was passed'.format(obj.__class__.__name__))

def split_list(l, num_parts):
    items = []
    leng = math.ceil(len(l) / float(num_parts))
    i = 0
    for item in l:
        if i >= leng:
            i = 0
        if i == 0:
            items.append([])
        if i < leng:
            items[-1].append(item)
        i += 1
    return items

def split_dict(d, num_parts):
    return [dict(l) for l in split_list(d.items(), num_parts)]
