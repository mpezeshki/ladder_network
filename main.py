import logging
import os
import time
import numpy
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
from datasets import get_streams
logger = logging.getLogger('main')


def setup_model():
    ladder = LadderAE()
    input_type = TensorType('float32', [False, False])
    x = input_type('features')
    y = theano.tensor.lvector('targets')
    ladder.apply(x, y)

    return ladder


def train(ladder, batch_size=100, num_train_examples=50000,
          num_epochs=150, lrate_decay=0.67):
    # Setting Logger
    timestr = time.strftime("%Y_%m_%d_at_%H_%M")
    save_path = 'results/mnist_' + timestr
    log_path = os.path.join(save_path, 'log.txt')
    os.makedirs(save_path)
    fh = logging.FileHandler(filename=log_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Training
    model = Model(ladder.costs.total)
    all_params = model.parameters
    print len(all_params)

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

    train_data_stream, valid_data_stream = get_streams(
        num_train_examples, batch_size)

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
        model=model,
        extensions=[
            train_monitoring,
            valid_monitoring,
            FinishAfter(after_n_epochs=num_epochs),
            SaveParams(None, all_params, save_path, after_epoch=True),
            SaveLog(save_path, after_epoch=True),
            LRDecay(lr=ladder.lr,
                    decay_first=num_epochs * lrate_decay,
                    decay_last=num_epochs,
                    after_epoch=True),
            Printing()])
    main_loop.run()


def evaluate(ladder, load_path):
    with open(load_path + '/trained_params.npz') as f:
        loaded = numpy.load(f)
        cg = ComputationGraph([ladder.costs.total])
        current_params = VariableFilter(roles=[PARAMETER])(cg.variables)
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
