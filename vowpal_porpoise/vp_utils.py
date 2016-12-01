import os
import sys
from multiprocessing import Pool
from platform import system
from retrying import retry
import os
import math
import socket

def get_os():
    platform = system()
    if 'Darwin' == platform:
        return 'Mac'
    else:
        return platform

class VPLogger:
    """
    Basic logger functionality; replace this with a real logger of your choice
    """
    def debug(self, s):
        print '[DEBUG] %s' % s

    def info(self, s):
        print '[INFO] %s' % s

    def warning(self, s):
        print '[WARNING] %s' % s

    def error(self, s):
        print '[ERROR] %s' % s


@retry(wait_random_min=1000, wait_random_max=2000, stop_max_attempt_number=40)
def netcat(hostname, port, content):
    print('Connecting to port {}'.format(port))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((hostname, port))
    s.sendall(content)
    s.shutdown(socket.SHUT_WR)
    data = []
    while True:
        datum = s.recv(1024)
        if datum == '':
            break
        datum = datum.split('\n')
        for dat in datum:
            if dat != '':
                dat = float(dat)
                if 1 >= dat >= -1:  #TODO: Parameterize
                    data.append(dat)
    s.close()
    return data


def vw_hash_process_key(key):
    if isinstance(key, list):
        if any(map(lambda x: isinstance(x, (list, dict)), key)):
            return ' '.join(map(vw_hash_process_key, key))
        return ' '.join(map(str, key))
    if isinstance(key, dict):
        return ' '.join([str(k) + ':' + str(v) for (k, v) in key.iteritems()])
    return str(key)

def vw_hash_to_vw_str(input_hash):
    vw_hash = input_hash.copy()
    vw_str = ''
    if vw_hash.get('label') is not None:
        vw_str += str(vw_hash.pop('label')) + ' '
        if vw_hash.get('importance'):
            vw_str += str(vw_hash.pop('importance')) + ' '
    return vw_str + ' '.join(['|' + k + ' ' + v for (k, v) in zip(vw_hash.keys(), map(vw_hash_process_key, vw_hash.values()))])
