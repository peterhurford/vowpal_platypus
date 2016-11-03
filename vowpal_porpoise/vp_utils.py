import os
import sys

def safe_remove(f):
    try:
        os.remove(f)
    except OSError:
        pass

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
