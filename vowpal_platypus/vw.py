from vp_utils import VPLogger, get_os, netcat, vw_hash_process_key
from multiprocessing import Pool
from contextlib import contextmanager
from random import randrange
from copy import deepcopy
import os
import sys
import subprocess
import shlex
import tempfile
import math
import collections

def is_list(x):
    return isinstance(x, collections.Sequence) and not isinstance(x, basestring)

def safe_remove(f):
    os.system('rm -r ' + str(f) + ' 2> /dev/null')

def split_file(filename, num_cores):
    if num_cores > 1:
        print('Splitting {}...'.format(filename))
        num_lines = sum(1 for line in open(filename))
        if get_os() == 'Mac':
            split = 'gsplit'
        else:
            split = 'split'
        os.system("{split} -d -l {lines} {filename} {filename}".format(split=split,
                                                                       lines=int(math.ceil(num_lines / float(num_cores))),
                                                                       filename=filename))
    else:
        os.system('cp {} {}00'.format(filename, filename))

def test_train_split(filename, train_pct=0.8, header=True):
    num_lines = sum(1 for line in open(filename)) - 1
    train_lines = int(math.ceil(num_lines * 0.8))
    test_lines = int(math.floor(num_lines * (1 - train_pct)))
    train_file = filename + 'train'
    test_file = filename + 'test'
    os.system('tail -n {} {} > {}'.format(num_lines, filename, filename + '_'))
    os.system('head -n {} {} > {}'.format(train_lines, filename + '_', train_file))
    os.system('head -n {} {} > {}'.format(test_lines, filename + '_', test_file))
    safe_remove(filename + '_')
    return (train_file, test_file)

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

class VW:
    def __init__(self, params):
        defaults = {'logger': None, 'vw': 'vw', 'name': 'VW', 'binary': False, 'link': None,
                    'bits': None, 'loss': None, 'passes': None, 'log_err': False, 'debug': False,
                    'debug_rate': 1000, 'l1': None, 'l2': None, 'learning_rate': None,
                    'quadratic': None, 'cubic': None, 'audit': None, 'power_t': None,
                    'adaptive': False, 'working_dir': None, 'decay_learning_rate': None,
                    'initial_t': None, 'lda': None, 'lda_D': None, 'lda_rho': None,
                    'lda_alpha': None, 'minibatch': None, 'total': None, 'node': None,
                    'holdout_off': False, 'threads': False, 'unique_id': None,
                    'span_server': None, 'bfgs': None, 'oaa': None, 'old_model': None,
                    'incremental': False, 'mem': None, 'nn': None, 'rank': None, 'lrq': None,
                    'lrqdropout': False, 'daemon': False, 'quiet': False, 'port': None,
                    'num_children': None}
        for param_name in params.keys():
            if param_name not in defaults.keys():
                raise ValueError('{} is not a supported VP parameter.'.format(param_name))
        self.params = params
        for (param_name, default_value) in defaults.iteritems():
            if default_value is not None:
                if self.params.get(param_name) is None:
                    self.params[param_name] = default_value

        assert self.params.get('name') is not None
        if not self.params.get('daemon'):
            assert self.params.get('passes') is not None

        self.log = self.params.get('logger')
        if self.log is None:
            self.log = VPLogger()

        if self.params.get('debug'):
            assert self.params.get('debug_rate') > 0
        if self.params.get('debug_rate') != 1000:
            self.params['debug'] = True

        if self.params.get('node') is not None:
            assert self.params.get('total') is not None
            assert self.params.get('unique_id') is not None
            assert self.params.get('span_server') is not None
            assert self.params.get('holdout_off')

        if self.params.get('daemon'):
            assert self.params.get('port') is not None
            assert self.params.get('node') is None

        self.handle = '%s' % self.params.get('name')
        if self.params.get('node') is not None:
            self.handle = "%s.%d" % (self.handle, self.params.get('node'))

        if self.params.get('old_model') is None:
            self.params['filename'] = '%s.model' % self.handle
            self.params['incremental'] = False
        else:
            self.params['filename'] = self.params['old_model']
            self.params['incremental'] = True

        if self.params.get('lda'):
            assert self.params.get('l1') is None
            assert self.params.get('l2') is None
            assert self.params.get('loss') is None
            assert self.params.get('adaptive') is None
            assert self.params.get('oaa') is None
            assert self.params.get('bfgs') is None
        else:
            assert self.params.get('lda_D') is None
            assert self.params.get('lda_rho') is None
            assert self.params.get('lda_alpha') is None
            assert self.params.get('minibatch') is None
        if self.params.get('lrqdropout') is None:
            assert self.params.get('lrq')

        self.working_directory = self.params.get('working_dir') or os.getcwd()

    def vw_base_command(self, base):
        l = base
        if self.params.get('bits')                is not None: l.append('-b ' + str(int(self.params['bits'])))
        if self.params.get('learning_rate')       is not None: l.append('--learning_rate ' + str(float(self.params['learning_rate'])))
        if self.params.get('l1')                  is not None: l.append('--l1 ' + str(float(self.params['l1'])))
        if self.params.get('l2')                  is not None: l.append('--l2 ' + str(float(self.params['l2'])))
        if self.params.get('initial_t')           is not None: l.append('--initial_t ' + str(float(self.params['initial_t'])))
        if self.params.get('binary'):                          l.append('--binary')
        if self.params.get('link')                is not None: l.append('--link ' + str(self.params['link']))
        if self.params.get('quadratic')           is not None: l.append(' '.join(['-q ' + str(s) for s in self.params['quadratic']]) if is_list(self.params['quadratic']) else '-q ' + str(self.params['quadratic']))
        if self.params.get('cubic')               is not None: l.append(' '.join(['--cubic ' + str(s) for s in self.params['cubic']]) if is_list(self.params['cubic']) else '--cubic ' + str(self.params['cubic']))
        if self.params.get('power_t')             is not None: l.append('--power_t ' + str(float(self.params['power_t'])))
        if self.params.get('loss')                is not None: l.append('--loss_function ' + str(self.params['loss']))
        if self.params.get('decay_learning_rate') is not None: l.append('--decay_learning_rate ' + str(float(self.params['decay_learning_rate'])))
        if self.params.get('lda')                 is not None: l.append('--lda ' + str(int(self.params['lda'])))
        if self.params.get('lda_D')               is not None: l.append('--lda_D ' + str(int(self.params['lda_D'])))
        if self.params.get('lda_rho')             is not None: l.append('--lda_rho ' + str(float(self.params['lda_rho'])))
        if self.params.get('lda_alpha')           is not None: l.append('--lda_alpha ' + str(float(self.params['lda_alpha'])))
        if self.params.get('minibatch')           is not None: l.append('--minibatch ' + str(int(self.params['minibatch'])))
        if self.params.get('oaa')                 is not None: l.append('--oaa ' + str(int(self.params['oaa'])))
        if self.params.get('unique_id')           is not None: l.append('--unique_id ' + str(int(self.params['unique_id'])))
        if self.params.get('total')               is not None: l.append('--total ' + str(int(self.params['total'])))
        if self.params.get('node')                is not None: l.append('--node ' + str(int(self.params['node'])))
        if self.params.get('threads'):                         l.append('--threads')
        if self.params.get('span_server')         is not None: l.append('--span_server ' + str(self.params['span_server']))
        if self.params.get('mem')                 is not None: l.append('--mem ' + str(int(self.params['mem'])))
        if self.params.get('audit'):                           l.append('--audit')
        if self.params.get('bfgs'):                            l.append('--bfgs')
        if self.params.get('adaptive'):                        l.append('--adaptive')
        if self.params.get('nn')                  is not None: l.append('--nn ' + str(int(self.params['nn'])))
        if self.params.get('rank')                is not None: l.append('--rank ' + str(int(self.params['rank'])))
        if self.params.get('lrq')                 is not None: l.append('--lrq ' + str(int(self.params['lrq'])))
        if self.params.get('lrqdropout'):                      l.append('--lrqdropout')
        if self.params.get('holdout_off'):                     l.append('--holdout_off')
        if self.params.get('quiet'):                           l.append('--quiet')
        return ' '.join(l)

    def vw_train_command(self, cache_file, model_file):
        if os.path.exists(model_file) and self.params['incremental']:
            return self.vw_base_command([self.params['vw']]) + ' --passes %d --cache_file %s -i %s -f %s' \
                    % (self.params['passes'], cache_file, model_file, model_file)
        else:
            return self.vw_base_command([self.params['vw']]) + ' --passes %d --cache_file %s -f %s' \
                    % (self.params['passes'], cache_file, model_file)

    def vw_test_command(self, model_file, prediction_file):
        l = [self.params['vw']]
        if self.params.get('threads'):                        l.append('--threads')
        if self.params.get('holdout_off'):                    l.append('--holdout_off')
        if self.params.get('quiet'):                          l.append('--quiet')
        if self.params.get('daemon'):
            print('Running a VW daemon on port %s' % self.params.get('port'))
            l.append('--daemon')
            if self.params.get('port') is not None:           l.append('--port ' + str(int(self.params['port'])))
            if self.params.get('num_children') is not None:   l.append('--num_children ' + str(int(self.params['num_children'])))
        cmd = ' '.join(l) + ' -t -i %s' % model_file
        if not self.params.get('daemon'):
            cmd += ' -p %s' % prediction_file
        return cmd

    def vw_test_command_library(self, model_file):
        return ' -t -i %s' % (model_file)

    def start_training(self):
        cache_file = self.get_cache_file()
        model_file = self.get_model_file()

        # Remove the old cache and model files
        if not self.params.get('incremental'):
            safe_remove(cache_file)
            safe_remove(model_file)

        # Run the actual training
        self.vw_process = self.make_subprocess(self.vw_train_command(cache_file, model_file))

        # set the instance pusher
        self.push_instance = self.push_instance_stdin

    def close_process(self):
        # Close the process
        assert self.vw_process
        self.vw_process.stdin.flush()
        self.vw_process.stdin.close()
        if self.params.get('port'):
            os.system("pkill -9 -f 'vw.*--port %i'" % self.params['port'])
        if self.vw_process.wait() != 0:
            raise Exception("vw_process %d (%s) exited abnormally with return code %d" % \
                (self.vw_process.pid, self.vw_process.command, self.vw_process.returncode))

    def push_instance_stdin(self, instance):
        vw_line = vw_hash_to_vw_str(instance)
        if self.params.get('debug') and randrange(0, self.params['debug_rate']) == 0:
            self.log.debug(vw_line)
        self.vw_process.stdin.write(('%s\n' % vw_line).encode('utf8'))

    def start_predicting(self):
        model_file = self.get_model_file()
        # Be sure that the prediction file has a unique filename, since many processes may try to
        # make predictions using the same model at the same time
        _, prediction_file = tempfile.mkstemp(dir='.', prefix=self.get_prediction_file())
        os.close(_)

        self.vw_process = self.make_subprocess(self.vw_test_command(model_file, prediction_file))
        self.prediction_file = prediction_file
        self.push_instance = self.push_instance_stdin

    @contextmanager
    def training(self):
        self.start_training()
        yield
        self.close_process()

    @contextmanager
    def predicting(self):
        self.start_predicting()
        yield
        self.close_process()


    def train_on(self, filename, line_function, evaluate_function=None, header=False):
        hyperparams = [k for (k, p) in self.params.iteritems() if is_list(p) and k not in ['quadratic', 'cubic']]
        if len(hyperparams):
            if evaluate_function is None:
                raise ValueError("evaluate_function must be defined in order to hypersearch.")
            num_lines = sum(1 for line in open(filename)) - 1
            train = int(math.ceil(num_lines * 0.8))
            test = int(math.floor(num_lines * 0.2))
            train_file = filename + '_vp_internal_train'
            test_file = filename + '_vp_internal_validate'
            os.system('head -n {} {} > {}'.format(train, filename, train_file))
            os.system('tail -n {} {} > {}'.format(test, filename, test_file))
            pos = 0
            for hyperparam in hyperparams:
                pos += 1
                hypermin, hypermax = self.params[hyperparam]
                if hypermax / float(hypermin) > 100:
                    param_range = [10 ** x for x in range(int(math.log10(hypermin)), int(math.log10(hypermax)) + 1)]
                else:
                    param_range = range(int(hypermin), int(hypermax) + 1)
                best_value = None
                best_metric = None
                model = deepcopy(self)
                model.params['quiet'] = True
                model.params['debug'] = False
                for other_hyperparam in hyperparams[pos:]:
                    average = (model.params[other_hyperparam][0] + model.params[other_hyperparam][1]) / 2.0
                    model.params[other_hyperparam] = average
                for value in param_range:
                    print('Trying {} as value for {}...'.format(value, hyperparam))
                    model.params[hyperparam] = value
                    model = model._run_train('{}_vp_internal_train'.format(filename), line_function=line_function, evaluate_function=None, header=header)
                    results = model.predict_on('{}_vp_internal_validate'.format(filename))
                    eval_metric = evaluate_function(results)
                    print('...{}'.format(eval_metric))
                    if best_metric is None or eval_metric > best_metric:  #TODO: <
                        best_metric = eval_metric
                        best_value = value
                print('Best value for {} was {}!'.format(hyperparam, best_value))
                self.params[hyperparam] = best_value
            os.system('rm {}_vp_internal_train'.format(filename))
            os.system('rm {}_vp_internal_validate'.format(filename))
            self.line_function = line_function
            self.evaluate_function = evaluate_function
            self.header = header
            safe_remove(train_file)
            safe_remove(test_file)
            return self
        else:
            return self._run_train(filename, line_function, evaluate_function, header)

    def _run_train(self, filename, line_function, evaluate_function, header):
        with self.training():
            with open(filename, 'r') as filehandle:
                if header:
                    filehandle.readline() # Throwaway header
                while True:
                    item = filehandle.readline()
                    if not item:
                        break
                    self.push_instance(line_function(item))
        self.line_function = line_function
        self.evaluate_function = evaluate_function
        self.header = header
        return self

    def predict_on(self, filename, line_function=None, evaluate_function=None, header=None):
        if line_function is None and self.line_function is not None:
            line_function = self.line_function
        if line_function is None:
            raise ValueError("A line function must be supplied for predicting.")
        if evaluate_function is None and self.evaluate_function is not None:
            evaluate_function = self.evaluate_function
        if header is None and self.header is not None:
            header = self.header
        else:
            header = False
        with self.predicting():
            actuals = []
            with open(filename, 'r') as filehandle:
                if header:
                    filehandle.readline() # Throwaway header
                while True:
                    item = filehandle.readline()
                    if not item:
                        break
                    item = line_function(item)
                    if item.get('label'):
                        actuals.append(item['label'])
                    self.push_instance(item)
        preds = self.read_predictions()
        if len(actuals) == len(preds):
            results = zip(preds, actuals)
            if evaluate_function is not None:
                print('Evaluated to: ' + str(evaluate_function(results)))
            return results
        else:
            return preds

    def parse_prediction(self, p):
        if self.params.get('lda'):
            return map(float, p.split())
        else:
            return float(p.split()[0])

    def read_predictions_(self):
        for x in open(self.prediction_file):
            yield self.parse_prediction(x)
        self.clean_predictions_file()

    def read_predictions(self):
        return list(self.read_predictions_())

    def clean_predictions_file(self):
        os.remove(self.prediction_file)

    def make_subprocess(self, command):
        if not self.params.get('log_err'):
            stdout = open('/dev/null', 'w')
            stderr = sys.stderr
            self.current_stdout = None
            self.current_stderr = None
        else:
            # Save the output of vw to file for debugging purposes
            log_file_base = tempfile.mktemp(dir=self.working_directory, prefix="vw-")
            self.current_stdout = log_file_base + '.out'
            self.current_stderr = log_file_base + '.err'
            stdout = open(self.current_stdout, 'w')
            stderr = open(self.current_stderr, 'w')
            stdout.write(command + '\n')
            stderr.write(command + '\n')
        
        if self.params.get('debug'):
            self.log.debug('Running command: "%s"' % str(command))
        result = subprocess.Popen(shlex.split(str(command)), stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, close_fds=True, universal_newlines=True)
        result.command = command
        return result

    def get_current_stdout(self):
        return open(self.current_stdout)

    def get_current_stderr(self):
        return open(self.current_stderr)

    def get_model_file(self):
        return os.path.join(self.working_directory, self.params['filename'])

    def get_cache_file(self):
        return os.path.join(self.working_directory, '%s.cache' % (self.handle))

    def get_prediction_file(self):
        return os.path.join(self.working_directory, '%s.prediction' % (self.handle))


def vw_model(model_params, node=False):
    default_params = {
        'name': 'VW',
        'unique_id': 0,
        'bits': 21
    }
    params = default_params
    params.update(model_params)
    if node is not False:
        multicore_params = {
            'total': model_params['cores'],
            'node': node,
            'holdout_off': True,
            'span_server': 'localhost'
        }
        params.update(multicore_params)
    if params.get('cores'):
        params.pop('cores')
    return VW(params)

def model(model_params):
    cores = model_params.get('cores')
    if cores is not None and cores > 1:
        return [vw_model(model_params, node=n) for n in range(cores)]
    else:
        return vw_model(model_params)

def linear_regression(**model_params):
    return model(model_params)

def als(**model_params):
    return model(model_params)

def logistic_regression(**model_params):
    model_params.update({'link': 'glf1', 'loss': 'logistic'})
    return model(model_params)

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

def vw_hash_to_vw_str(input_hash):
    vw_hash = input_hash.copy()
    vw_str = ''
    if vw_hash.get('label') is not None:
        vw_str += str(vw_hash.pop('label')) + ' '
        if vw_hash.get('importance'):
            vw_str += str(vw_hash.pop('importance')) + ' '
    return vw_str + ' '.join(['|' + str(k) + ' ' + str(v) for (k, v) in zip(vw_hash.keys(), map(vw_hash_process_key, vw_hash.values()))])

def daemon_predict(daemon, content, quiet=False):
    return netcat('localhost',
                  port=daemon.params['port'],
                  content=content,
                  quiet=daemon.params['quiet'] or quiet)


def run_(model, filename, line_function=None, train_line_function=None, predict_line_function=None, evaluate_function=None, split=0.8, header=True):
    train_file, test_file = test_train_split(filename, train_pct=split, header=header)
    if is_list(model):
        model = model[0]
    results = (model.train_on(train_file,
                              line_function=train_line_function,
                              evaluate_function=evaluate_function)
                     .predict_on(test_file,
                                 line_function=predict_line_function))
    safe_remove(train_file)
    safe_remove(test_file)
    safe_remove(model.get_cache_file())
    return results

def run_model(args):
    return run_(**args)

def run(model, filename, line_function=None, train_line_function=None, predict_line_function=None, evaluate_function=None, split=0.8, header=True):
    if train_line_function is None and line_function is not None:
        train_line_function = line_function
    if predict_line_function is None and line_function is not None:
        predict_line_function = line_function
    num_cores = len(model) if isinstance(model, collections.Sequence) else 1
    if num_cores > 1:
        os.system("spanning_tree")
        split_file(filename, num_cores)
        pool = Pool(num_cores)
        filenames = [filename + (str(n) if n >= 10 else '0' + str(n)) for n in range(num_cores)]
        args = []
        for i in range(num_cores):
            args.append({'model': model[i],
                         'filename': filenames[i],
                         'train_line_function': train_line_function,
                         'predict_line_function': predict_line_function,
                         'split': split,
                         'header': header})
        results = sum(pool.map(run_model, args), [])
        if evaluate_function:
            print(evaluate_function(results))
        for f in filenames:
            safe_remove(f)
        os.system('killall spanning_tree')
    else:
        return run_(model,
                    filename=filename,
                    train_line_function=train_line_function,
                    predict_line_function=predict_line_function,
                    evaluate_function=evaluate_function,
                    split=split,
                    header=header)
