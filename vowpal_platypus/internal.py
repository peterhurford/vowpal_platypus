import os
import sys
from multiprocessing import Pool
from platform import system
from retrying import retry
import os
import math
import socket
import string

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


@retry(wait_random_min=1000, wait_random_max=2000, stop_max_attempt_number=4)
def netcat(hostname, port, content, quiet=False):
    if not quiet:
        print('Connecting to port {}'.format(port))
    s = socket.socket()
    s.connect((hostname, port))
    s.sendall(content.encode('utf-8'))
    s.shutdown(socket.SHUT_WR)
    data = []
    while True:
        datum = s.recv(16384)
        if datum == '':
            break
        datum = datum.split('\n')
        for dat in datum:
            if dat != '':
                data.append(float(dat))
    s.close()
    return data


def to_str(s):
    """Convert to a string if it is not already a string. Useful for working around unicode issues."""
    if isinstance(s, basestring):
        return s
    else:
        return str(s)

def vw_hash_process_key(key):
    if isinstance(key, list):
        if any(map(lambda x: isinstance(x, (list, dict)), key)):
            return ' '.join(map(vw_hash_process_key, key))
        return ' '.join(map(to_str, key))
    if isinstance(key, dict):
        if not all(map(lambda x: isinstance(x, int) or isinstance(x, float), key.values())):
            raise ValueError('Named values passed to VP must be numeric.')
        if not all(map(lambda x: isinstance(x, basestring), key.keys())):
            raise ValueError('Named values passed to VP must have strings for names.')
        return ' '.join([to_str(k) + ':' + to_str(v) for (k, v) in key.iteritems()])
    return to_str(key)
