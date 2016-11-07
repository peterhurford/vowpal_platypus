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

@retry(wait_fixed=1000, stop_max_attempt_number=10)
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
                if 1 >= dat >= 0:  #TODO: Parameterize
                    data.append(dat)
    s.close()
    return data
