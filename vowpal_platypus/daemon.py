from internal import netcat
from vw import VW

def daemon(model):
    if model.params.get('node'):
        port = model.params['node'] + 4040
    else:
        port = 4040
    daemon_model = VW({'name': model.handle,
                       'daemon': True,
                       'old_model': model.get_model_file(),
                       'holdout_off': True,
                       'quiet': True if model.params.get('quiet') else False,
                       'port': port,
                       'num_children': 2})
    return daemon_model

def daemon_predict(daemon, content, quiet=False):
    return netcat('localhost',
                  port=daemon.params['port'],
                  content=content,
                  quiet=daemon.params['quiet'] or quiet)
