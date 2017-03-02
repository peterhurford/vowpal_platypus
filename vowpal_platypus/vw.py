from internal import VPLogger
from utils import safe_remove, shuffle_file, split_file, vw_hash_to_vw_str

import os
import sys
import subprocess
import shlex
import tempfile
import math
import collections
import traceback

from pathos.multiprocessing import ProcessingPool as Pool
from contextlib import contextmanager
from random import randrange
from copy import deepcopy


class VW:
    def __init__(self, params):
        defaults = {'logger': None, 'vw': 'vw', 'name': 'VW', 'binary': False, 'link': None,
                    'bits': 21, 'loss': None, 'passes': 1, 'log_err': False, 'debug': False,
                    'debug_rate': 1000, 'l1': None, 'l2': None, 'learning_rate': None,
                    'quadratic': None, 'cubic': None, 'audit': None, 'power_t': None,
                    'adaptive': False, 'working_dir': None, 'decay_learning_rate': None,
                    'initial_t': None, 'lda': None, 'lda_D': None, 'lda_rho': None,
                    'lda_alpha': None, 'minibatch': None, 'total': None, 'node': None,
                    'holdout_off': False, 'threads': False, 'unique_id': None,
                    'span_server': None, 'bfgs': None, 'termination': None, 'oaa': None, 'old_model': None,
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

        assert self.params.get('name') is not None, 'A VP model must have a name.'
        assert isinstance(self.params['name'], basestring), 'The VP model must be a string.'
        assert ' ' not in self.params['name'], 'A VP model name cannot contain a space character.'
        if not self.params.get('daemon'):
            assert self.params.get('passes') is not None, 'Please specify a value for number of passes.'

        self.log = self.params.get('logger')
        if self.log is None:
            self.log = VPLogger()

        if self.params.get('debug'):
            assert self.params.get('debug_rate') > 0, 'Debugging requires a debug rate greater than 0.'
        if self.params.get('debug_rate') != 1000:
            self.params['debug'] = True

        if self.params.get('node') is not None:
            assert self.params.get('total') is not None, 'Please specify the total number of nodes in your cluster.'
            assert self.params.get('unique_id') is not None, 'A VP cluster requires a unique id.'
            assert self.params.get('span_server') is not None, 'Please specify the location of your VP cluster span server.'
            assert self.params.get('holdout_off'), 'VP clusters do not work with holdout sets. Please specify `holdout_off`.'

        if self.params.get('daemon'):
            assert self.params.get('port') is not None, 'Please specify a port for your VP daemon.'
            assert self.params.get('node') is None, 'Your VP daemon cannot run in a cluster.'

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
            assert self.params.get('l1') is None, 'L1 does not work in LDA mode.'
            assert self.params.get('l2') is None, 'L2 does not work in LDA mode.'
            assert self.params.get('loss') is None, 'You cannot specify a loss with LDA mode.'
            assert self.params.get('adaptive') is None, 'Adaptive mode is not compatible with LDA mode.'
            assert self.params.get('oaa') is None, '`oaa` is not compatible with LDA mode.'
            assert self.params.get('bfgs') is None, '`bfgs` is not compatible with LDA mode.'
            assert self.params.get('termination') is None, '`termination` is not compatible with LDA mode.'
        else:
            assert self.params.get('lda_D') is None, '`lda_d` parameter requires LDA mode.'
            assert self.params.get('lda_rho') is None, '`lda_rho` parameter requires LDA mode.'
            assert self.params.get('lda_alpha') is None, '`lda_alpha` parameter requires LDA mode.'
            assert self.params.get('minibatch') is None, '`minibatch` parameter requires LDA mode.'
        if self.params.get('lrqdropout') is None:
            assert self.params.get('lrq'), '`lrqdropout` parameter requires an `lrq` parameter'

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
        if self.params.get('quadratic')           is not None: l.append(' '.join(['-q ' + str(s) for s in self.params['quadratic']]) if isinstance(self.params['quadratic'], list) else '-q ' + str(self.params['quadratic']))
        if self.params.get('cubic')               is not None: l.append(' '.join(['--cubic ' + str(s) for s in self.params['cubic']]) if isinstance(self.params['cubic'], list) else '--cubic ' + str(self.params['cubic']))
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
        if self.params.get('termination'):                     l.append('--termination ' + str(int(self.params['termination'])))
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
        logistic = self.params.get('loss') == 'logistic'
        vw_line = vw_hash_to_vw_str(instance, logistic=logistic)
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


    def train_on(self, filename, line_function, evaluate_function=None, header=True):
        hyperparams = [k for (k, p) in self.params.iteritems() if isinstance(p, list) and k not in ['quadratic', 'cubic']]
        if len(hyperparams):
            if evaluate_function is None:
                raise ValueError("evaluate_function must be defined in order to hypersearch.")
            num_lines = sum(1 for line in open(filename))
            train = int(math.ceil(num_lines * 0.8))
            test = int(math.floor(num_lines * 0.2))
            train_file = filename + '_vp_hypersearch_train'
            test_file = filename + '_vp_hypersearch_validate'
            filename = shuffle_file(filename, header=header)
            os.system('head -n {} {} > {}'.format(train, filename, train_file))
            os.system('tail -n {} {} > {}'.format(test, filename, test_file))
            pos = 0
            for hyperparam in hyperparams:
                pos += 1
                if len(self.params[hyperparam]) == 2:
                    hypermin, hypermax = self.params[hyperparam]
                    if hypermax / float(hypermin) > 100:
                        param_range = [10 ** x for x in range(int(math.log10(hypermin)), int(math.log10(hypermax)) + 1)]
                    else:
                        param_range = range(int(hypermin), int(hypermax) + 1)
                else:
                    param_range = self.params[hyperparam]
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
                    model = model._run_train(train_file, line_function=line_function, evaluate_function=None, header=header)
                    results = model.predict_on(test_file)
                    eval_metric = evaluate_function(results)
                    print('...{}'.format(eval_metric))
                    if best_metric is None or eval_metric < best_metric:  #TODO: >
                        best_metric = eval_metric
                        best_value = value
                print('Best value for {} was {}!'.format(hyperparam, best_value))
                self.params[hyperparam] = best_value
            self.line_function = line_function
            self.evaluate_function = evaluate_function
            self.header = header
            safe_remove(train_file)
            safe_remove(test_file)
            safe_remove(filename)
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
        elif header is None:
            header = True
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
                    if item.get('label') is not None:
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
        safe_remove(self.prediction_file)

    def read_predictions(self):
        return list(self.read_predictions_())

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


def test_train_split(filename, train_pct=0.8, header=True):
    num_lines = sum(1 for line in open(filename)) - 1
    train_lines = int(math.ceil(num_lines * 0.8))
    test_lines = int(math.floor(num_lines * (1 - train_pct)))
    filename = shuffle_file(filename, header=header)
    train_file = filename + 'train'
    test_file = filename + 'test'
    os.system('tail -n {} {} > {}'.format(num_lines, filename, filename + '_'))
    os.system('head -n {} {} > {}'.format(train_lines, filename + '_', train_file))
    os.system('head -n {} {} > {}'.format(test_lines, filename + '_', test_file))
    safe_remove(filename + '_')
    safe_remove(filename)
    return (train_file, test_file)

def run_(model, train_filename=None, predict_filename=None, train_line_function=None, predict_line_function=None, evaluate_function=None, split=0.8, header=True, quiet=False, multicore=False):
    if isinstance(model, list):
        model = model[0]
    if train_filename == predict_filename:
        train_filename, predict_filename = test_train_split(train_filename, train_pct=split, header=header)
    results = (model.train_on(train_filename,
                              line_function=train_line_function,
                              evaluate_function=evaluate_function,
                              header=header)
                     .predict_on(predict_filename,
                                 line_function=predict_line_function,
                                 header=header))
    if not quiet and multicore:
        print 'Shuffling...'
    if train_filename == predict_filename:
        safe_remove(train_filename)
        safe_remove(predict_filename)
    safe_remove(model.get_cache_file())
    safe_remove(model.get_model_file())
    return results

def run_model(args):
    return run_(**args)

def run(model, filename=None, train_filename=None, predict_filename=None, line_function=None, train_line_function=None, predict_line_function=None, evaluate_function=None, split=0.8, header=True):
    if train_line_function is None and line_function is not None:
        train_line_function = line_function
    if predict_line_function is None and line_function is not None:
        predict_line_function = line_function
    if train_filename is None and filename is not None:
        train_filename = filename
    if predict_filename is None and filename is not None:
        predict_filename = filename
    num_cores = len(model) if isinstance(model, collections.Sequence) else 1
    if num_cores > 1:
        os.system("spanning_tree")
        if header:
            num_lines = sum(1 for line in open(train_filename))
            os.system('tail -n {} {} > {}'.format(num_lines - 1, train_filename, train_filename + '_'))
            if predict_filename != train_filename:
                num_lines = sum(1 for line in open(predict_filename))
                os.system('tail -n {} {} > {}'.format(num_lines - 1, predict_filename, predict_filename + '_'))
            train_filename = train_filename + '_'
            predict_filename = predict_filename + '_'
            header = False
        split_file(train_filename, num_cores)
        if predict_filename != train_filename:
            split_file(predict_filename, num_cores)
        pool = Pool(num_cores)
        train_filenames = [train_filename + (str(n) if n >= 10 else '0' + str(n)) for n in range(num_cores)]
        predict_filenames = [predict_filename + (str(n) if n >= 10 else '0' + str(n)) for n in range(num_cores)]
        args = []
        for i in range(num_cores):
            args.append({'model': model[i],
                         'train_filename': train_filenames[i],
                         'predict_filename': predict_filenames[i],
                         'train_line_function': train_line_function,
                         'predict_line_function': predict_line_function,
                         'evaluate_function': evaluate_function,
                         'split': split,
                         'quiet': model[i].params.get('quiet'),
                         'multicore': True,
                         'header': header})
        results = sum(pool.map(run_model, args), [])
        if evaluate_function:
            print(evaluate_function(results))
        for f in train_filenames + predict_filenames:
            safe_remove(f)
        os.system('killall spanning_tree')
        return results
    else:
        return run_(model,
                    train_filename=train_filename,
                    predict_filename=predict_filename,
                    train_line_function=train_line_function,
                    predict_line_function=predict_line_function,
                    evaluate_function=evaluate_function,
                    split=split,
                    quiet=model.params.get('quiet'),
                    multicore=False,
                    header=header)


def run_parallel(vw_models, core_fn, spindown=True):
    num_cores = len(vw_models) if isinstance(vw_models, collections.Sequence) else 1
    if num_cores > 1:
        os.system("spanning_tree")
        def run_fn(model):
            try:
                return core_fn(model)
            except Exception as e:
                print('ERROR: Caught exception in worker thread (x = %d):' % model.params['node'])
                traceback.print_exc()
                #os.system('killall vw')
                #os.system('killall spanning_tree')
                raise e
        pool = Pool(num_cores)
        results = pool.map(run_fn, vw_models) # TODO: Integrate into `run`
        if spindown:
            os.system('killall vw')
            os.system('killall spanning_tree')
        return results
    else:
        return core_fn(vw_models[0] if isinstance(vw_models, collections.Sequence) else vw_models)
