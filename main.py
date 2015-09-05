import logging
import os
import numpy
import time
import theano
from theano.tensor.type import TensorType
from blocks.algorithms import GradientDescent, Adam
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import PARAMETER
from utils import SaveLog, SaveParams, LRDecay
from ladder import LadderAE
from datasets import make_datastream
from datasets import get_mnist_data_dict
logger = logging.getLogger('main')


def setup_model():
    ladder = LadderAE()
    input_type = TensorType('float32', [False, False])
    x_only = input_type('features_unlabeled')
    x = input_type('features_labeled')
    y = theano.tensor.lvector('targets_labeled')
    ladder.apply(x, y, x_only)

    return ladder


def train(ladder, batch_size=100, labeled_samples=100,
          unlabeled_samples=50000, valid_set_size=10000,
          num_epochs=150, valid_batch_size=100, lrate_decay=0.67,
          save_path='results/mnist_100_full0'):
    # Setting Logger
    log_path = os.path.join(save_path, 'log.txt')
    fh = logging.FileHandler(filename=log_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info('Logging into %s' % log_path)

    # Training
    all_params = ComputationGraph([ladder.costs.total]).parameters
    logger.info('Found the following parameters: %s' % str(all_params))

    training_algorithm = GradientDescent(
        cost=ladder.costs.total, params=all_params,
        step_rule=Adam(learning_rate=ladder.lr))

    # Fetch all batch normalization updates. They are in the clean path.
    # In addition to actual training, also do BN variable approximations
    bn_updates = ComputationGraph([ladder.costs.class_clean]).updates
    training_algorithm.add_updates(bn_updates)

    monitored_variables = [
        ladder.costs.class_corr, ladder.costs.class_clean,
        ladder.error, training_algorithm.total_gradient_norm,
        ladder.costs.total] + ladder.costs.denois.values()

    data = get_mnist_data_dict(unlabeled_samples=unlabeled_samples,
                               valid_set_size=valid_set_size)

    train_data_stream = make_datastream(
        data.train, data.train_ind, batch_size,
        n_labeled=labeled_samples,
        n_unlabeled=unlabeled_samples)

    valid_data_stream = make_datastream(
        data.valid, data.valid_ind, valid_batch_size,
        n_labeled=len(data.valid_ind),
        n_unlabeled=len(data.valid_ind))

    train_monitoring = TrainingDataMonitoring(
        variables=monitored_variables,
        prefix="train",
        after_epoch=True)

    valid_monitoring = DataStreamMonitoring(
        variables=monitored_variables,
        data_stream=valid_data_stream,
        prefix="valid",
        after_epoch=True)

    main_loop = MainLoop(
        algorithm=training_algorithm,
        data_stream=train_data_stream,
        model=Model(ladder.costs.total),
        extensions=[
            train_monitoring,
            valid_monitoring,
            FinishAfter(after_n_epochs=num_epochs),
            SaveParams(None, all_params, save_path, after_epoch=True),
            SaveLog(save_path, after_training=True),
            LRDecay(lr=ladder.lr,
                    decay_first=num_epochs * lrate_decay,
                    decay_last=num_epochs,
                    after_epoch=True),
            Printing()])
    main_loop.run()


def evaluate(ladder, load_path):
    logger.info('Evaluating pretrained model')
    with open(load_path + '/trained_params.npz') as f:
        loaded = numpy.load(f)
        cg = ComputationGraph([ladder.costs.total])
        current_params = VariableFilter(roles=[PARAMETER])(cg.variables)
        logger.info('Loading parameters: %s' % ', '.join(loaded.keys()))
        for param in current_params:
            assert param.get_value().shape == loaded[param.name].shape
            param.set_value(loaded[param.name])


if __name__ == "__main__":
    load_path = None
    t_start = time.time()
    logging.basicConfig(level=logging.INFO)
    ladder = setup_model()
    if load_path is None:
        train(ladder)
    else:
        evaluate(ladder, load_path)
    logger.info('Took %.1f minutes' % ((time.time() - t_start) / 60.))
