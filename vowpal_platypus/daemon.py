from internal import netcat
from utils import vw_hash_to_vw_str
from vw import VW

def daemon(model, port=4040):
    if model.params.get('node'):
        port = model.params['node'] + port
    daemon_model = VW({'name': model.handle,
                       'daemon': True,
                       'old_model': model.get_model_file(),
                       'holdout_off': True,
                       'quiet': True,
                       'port': port,
                       'num_children': 2})
    daemon_model.start_predicting()
    return daemon_model

def daemon_predict(daemon, content, quiet=False):
    if len(content) == 1:
        content = [content]
    if isinstance(content[0], dict):
        content = '\n'.join(map(vw_hash_to_vw_str, content))
    return netcat('localhost',
                  port=daemon.params['port'],
                  content=content,
                  quiet=daemon.params['quiet'] or quiet)
