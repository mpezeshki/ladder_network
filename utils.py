import os
import logging
import numpy as np
import theano
import theano.tensor as T
from blocks.extensions import SimpleExtension
from blocks.roles import add_role
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


class AttributeDict(dict):
    __getattr__ = dict.__getitem__

    def __setattr__(self, a, b):
        self.__setitem__(a, b)


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
        np.savez_compressed(path, **self.to_save)

    def do(self, which_callback, *args):
        if self.var_name is None:
            return
        val = self.main_loop.log.current_row[self.var_name]
        if self.best_value is None or val < self.best_value:
            self.best_value = val
        self.to_save = {v.name: v.get_value() for v in self.params}


class SaveLog(SimpleExtension):
    def __init__(self, dir, show=None, **kwargs):
        super(SaveLog, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        epoch = self.main_loop.status['epochs_done']
        current_row = self.main_loop.log.current_row
        logger.info("\nIter:%d" % epoch)
        for element in current_row:
            logger.info(str(element) + ":%f" % current_row[element])


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
