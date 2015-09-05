import os
import logging
import numpy as np
import theano
import theano.tensor as T
from pandas import DataFrame, read_hdf
from blocks.extensions import Printing, SimpleExtension
from blocks.main_loop import MainLoop
from blocks.roles import add_role
import functools
from argparse import ArgumentParser, Action
from blocks.roles import AuxiliaryRole

logger = logging.getLogger('main.utils')


class BnParamRole(AuxiliaryRole):
    pass
BNPARAM = BnParamRole()


def shared_param(init, name, cast_float32, role, **kwargs):
    if cast_float32:
        v = np.float32(init)
    p = theano.shared(v, name=name, **kwargs)
    add_role(p, role)
    return p


class LRDecay(SimpleExtension):
    def __init__(self, lr, decay_first, decay_last, **kwargs):
        super(LRDecay, self).__init__(**kwargs)
        self.iter = 0
        self.decay_first = decay_first
        self.decay_last = decay_last
        self.lr = lr
        self.lr_init = lr.get_value()

    def do(self, which_callback, *args):
        self.iter += 1
        if self.iter > self.decay_first:
            ratio = 1.0 * (self.decay_last - self.iter)
            ratio = np.maximum(0, ratio / (self.decay_last - self.decay_first))
            self.lr.set_value(np.float32(ratio * self.lr_init))
        logger.info("Iter %d, lr %f" % (self.iter, self.lr.get_value()))


class AttributeDict(dict):
    __getattr__ = dict.__getitem__

    def __setattr__(self, a, b):
        self.__setitem__(a, b)


class DummyLoop(MainLoop):
    def __init__(self, extensions):
        return super(DummyLoop, self).__init__(algorithm=None,
                                               data_stream=None,
                                               extensions=extensions)

    def run(self):
        for extension in self.extensions:
            extension.main_loop = self
        self._run_extensions('before_training')
        self._run_extensions('after_training')


class ShortPrinting(Printing):
    def __init__(self, to_print, use_log=True, **kwargs):
        self.to_print = to_print
        self.use_log = use_log
        super(ShortPrinting, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        log = self.main_loop.log

        # Iteration
        msg = "e {}, i {}:".format(
            log.status['epochs_done'],
            log.status['iterations_done'])

        # Requested channels
        items = []
        for k, vars in self.to_print.iteritems():
            for shortname, vars in vars.iteritems():
                if vars is None:
                    continue
                if type(vars) is not list:
                    vars = [vars]

                s = ""
                for var in vars:
                    try:
                        name = k + '_' + var.name
                        val = log.current_row[name]
                    except:
                        continue
                    try:
                        s += ' ' + ' '.join(["%.3g" % v for v in val])
                    except:
                        s += " %.3g" % val
                if s != "":
                    items += [shortname + s]
        msg = msg + ", ".join(items)
        if self.use_log:
            logger.info(msg)
        else:
            print msg


class SaveParams(SimpleExtension):
    """Finishes the training process when triggered."""
    def __init__(self, trigger_var, params, save_path, **kwargs):
        super(SaveParams, self).__init__(**kwargs)
        if trigger_var is None:
            self.var_name = None
        else:
            self.var_name = trigger_var[0] + '_' + trigger_var[1].name
        self.save_path = save_path
        self.params = params
        self.to_save = {}
        self.best_value = None
        self.add_condition('after_training', self.save)
        self.add_condition('on_interrupt', self.save)

    def save(self, which_callback, *args):
        if self.var_name is None:
            self.to_save = {v.name: v.get_value() for v in self.params}
        path = self.save_path + '/trained_params'
        logger.info('Saving to %s' % path)
        np.savez_compressed(path, **self.to_save)

    def do(self, which_callback, *args):
        if self.var_name is None:
            return
        val = self.main_loop.log.current_row[self.var_name]
        if self.best_value is None or val < self.best_value:
            self.best_value = val
        self.to_save = {v.name: v.get_value() for v in self.params}


class SaveArgs(SimpleExtension):
    def __init__(self, args, **kwargs):
        dir = args.save_dir
        super(SaveArgs, self).__init__(**kwargs)
        self.dir = dir
        self.args = args

    def do(self, which_callback, *args):
        df = DataFrame.from_dict(self.args, orient='index')
        df.to_hdf(os.path.join(self.dir, 'params'), 'params', mode='w',
                  complevel=5, complib='blosc')


class SaveLog(SimpleExtension):
    def __init__(self, dir, show=None, **kwargs):
        super(SaveLog, self).__init__(**kwargs)
        self.dir = dir
        self.show = show if show is not None else []

    def do(self, which_callback, *args):
        df = self.main_loop.log.to_dataframe()
        df.to_hdf(os.path.join(self.dir, 'log'), 'log', mode='w',
                  complevel=5, complib='blosc')


def prepare_dir(save_to, results_dir='results'):
    base = os.path.join(results_dir, save_to)
    i = 0

    while True:
        name = base + str(i)
        try:
            os.makedirs(name)
            break
        except:
            i += 1

    return name


def load_df(dirpath, filename, varname=None):
    varname = filename if varname is None else varname
    fn = os.path.join(dirpath, filename)
    return read_hdf(fn, varname)


def filter_funcs_prefix(d, pfx):
    pfx = 'cmd_'
    fp = lambda x: x.find(pfx)
    return {n[fp(n) + len(pfx):]: v for n, v in d.iteritems() if fp(n) >= 0}


def args_parser():
    rep = lambda s: s.replace('-', ',')
    chop = lambda s: s.split(',')
    to_int = lambda ss: [int(s) for s in ss if s.isdigit()]
    to_float = lambda ss: [float(s) for s in ss]

    def to_bool(s):
        if s.lower() in ['true', 't']:
            return True
        elif s.lower() in ['false', 'f']:
            return False
        else:
            raise Exception("Unknown bool value %s" % s)

    def compose(*funs):
        return functools.reduce(lambda f, g: lambda x: f(g(x)), funs)

    # Functional parsing logic to allow flexible function compositions
    # as actions for ArgumentParser
    def funcs(additional_arg):
        class customAction(Action):
            def __call__(self, parser, args, values, option_string=None):

                def process(arg, func_list):
                    if arg is None:
                        return None
                    elif type(arg) is list:
                        return map(compose(*func_list), arg)
                    else:
                        return compose(*func_list)(arg)

                setattr(args, self.dest, process(values, additional_arg))
        return customAction

    ap = ArgumentParser("Semisupervised experiment")
    subparsers = ap.add_subparsers(dest='cmd', help='sub-command help')

    # TRAIN
    train_cmd = subparsers.add_parser('train', help='Train a new model')
    parser = train_cmd

    a = parser.add_argument

    # General hyper parameters and settings
    a("save_to", help="Destination to save the state and results",
      default="noname", nargs="?")
    a("--num-epochs", help="Number of training epochs",
      type=int, default=150)
    a("--seed", help="Seed",
      type=int, default=[1], nargs='+')
    a("--dseed", help="Data permutation seed, defaults to 'seed'",
      type=int, default=[None], nargs='+')
    a("--labeled-samples", help="How many supervised samples are used",
      type=int, default=None, nargs='+')
    a("--unlabeled-samples", help="How many unsupervised samples are used",
      type=int, default=None, nargs='+')
    a("--dataset", type=str, default=['mnist'], nargs='+',
      choices=['mnist', 'cifar10'], help="Which dataset to use")
    a("--lr", help="Initial learning rate",
      type=float, default=[0.002], nargs='+')
    a("--lrate-decay", help="When to linearly start decaying lrate (0-1)",
      type=float, default=[0.67], nargs='+')
    a("--batch-size", help="Minibatch size",
      type=int, default=[100], nargs='+')
    a("--valid-batch-size", help="Minibatch size for validation data",
      type=int, default=[100], nargs='+')
    a("--valid-set-size", help="Number of examples in validation set",
      type=int, default=[10000], nargs='+')

    # Hyperparameters controlling supervised path
    a("--super-noise-std", help="Noise added to supervised learning path",
      type=float, default=[0.3], nargs='+')
    a("--f-local-noise-std", help="Noise added encoder path",
      type=str, default=[0.3], nargs='+',
      action=funcs([tuple, to_float, chop]))
    a("--act", nargs='+', type=str, action=funcs([tuple, chop, rep]),
      default=["relu"], help="List of activation functions")
    a("--encoder-layers", help="List of layers for f",
      type=str, default=(), action=funcs([tuple, chop, rep]))

    # Hyperparameters controlling unsupervised training
    a("--denoising-cost-x", help="Weight of the denoising cost.",
      type=str, default=[(0.,)], nargs='+',
      action=funcs([tuple, to_float, chop]))
    a("--decoder-spec", help="List of decoding function types",
      type=str, default=['sig'], action=funcs([tuple, chop, rep]))

    # Hyperparameters used for Cifar training
    a("--contrast-norm", help="Scale of contrast normalization (0=off)",
      type=int, default=[0], nargs='+')
    a("--top-c", help="Have c at softmax?", action=funcs([to_bool]),
      default=[True], nargs='+')
    a("--whiten-zca", help="Whether to whiten the data with ZCA",
      type=int, default=[0], nargs='+')

    # EVALUATE
    load_cmd = subparsers.add_parser('evaluate', help='Evaluate test error')
    load_cmd.add_argument('load_from', type=str,
                          help="Destination to load the state from")
    load_cmd.add_argument('--data-type', type=str, default='test',
                          help="Data set to evaluate on")

    args = ap.parse_args()
    return args


def apply_act(input, act_name):
    if input is None:
        return input
    act = {
        'relu': lambda x: T.maximum(0, x),
        'leakyrelu': lambda x: T.switch(x > 0., x, 0.1 * x),
        'linear': lambda x: x,
        'softplus': lambda x: T.log(1. + T.exp(x)),
        'sigmoid': lambda x: T.nnet.sigmoid(x),
        'softmax': lambda x: T.nnet.softmax(x),
    }.get(act_name)
    if act_name == 'softmax':
        input = input.flatten(2)
    return act(input)
