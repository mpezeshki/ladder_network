import logging
import os
import time
import numpy as np
import theano
from theano.tensor.type import TensorType
from blocks.algorithms import GradientDescent, Adam
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from utils import SaveLog, SaveParams, LRDecay
from ladder import LadderAE
from datasets import get_mixed_streams
logger = logging.getLogger('main')


def setup_model():
    ladder = LadderAE()
    input_type = TensorType('float32', [False, False])
    x_lb = input_type('features_labeled')
    x_un = input_type('features_unlabeled')
    y = theano.tensor.lvector('targets_labeled')
    ladder.apply(x_lb, x_un, y)

    return ladder


def train(ladder, batch_size=100, num_train_examples=60000,
          num_epochs=150, lrate_decay=0.67):
    # Setting Logger
    timestr = time.strftime("%Y_%m_%d_at_%H_%M")
    save_path = 'results/mnist_100_standard_' + timestr
    log_path = os.path.join(save_path, 'log.txt')
    os.makedirs(save_path)
    fh = logging.FileHandler(filename=log_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Training
    model = Model(ladder.costs.total)
    all_params = model.parameters
    print len(all_params)
    print all_params

    training_algorithm = GradientDescent(
        cost=ladder.costs.total, parameters=all_params,
        step_rule=Adam(learning_rate=ladder.lr))

    # Fetch all batch normalization updates. They are in the clean path.
    # In addition to actual training, also do BN variable approximations
    bn_updates = ComputationGraph([ladder.costs.class_clean]).updates
    training_algorithm.add_updates(bn_updates)

    monitored_variables = [
        ladder.costs.class_corr, ladder.costs.class_clean,
        ladder.error, training_algorithm.total_gradient_norm,
        ladder.costs.total] + ladder.costs.denois.values()

    train_data_stream, test_data_stream = get_mixed_streams(
        batch_size)

    train_monitoring = TrainingDataMonitoring(
        variables=monitored_variables,
        prefix="train",
        after_epoch=True)

    valid_monitoring = DataStreamMonitoring(
        variables=monitored_variables,
        data_stream=test_data_stream,
        prefix="test",
        after_epoch=True)

    main_loop = MainLoop(
        algorithm=training_algorithm,
        data_stream=train_data_stream,
        model=model,
        extensions=[
            train_monitoring,
            valid_monitoring,
            FinishAfter(after_n_epochs=num_epochs),
            SaveParams('test_CE_corr', model, save_path),
            SaveLog(save_path, after_epoch=True),
            LRDecay(lr=ladder.lr,
                    decay_first=num_epochs * lrate_decay,
                    decay_last=num_epochs,
                    after_epoch=True),
            Printing()])
    main_loop.run()


def evaluate(ladder, load_path):
    with open(load_path + '/trained_params_best.npz') as f:
        loaded = np.load(f)
        model = Model(ladder.costs.total)
        params_dicts = model.get_parameter_dict()
        params_names = params_dicts.keys()
        for param_name in params_names:
            param = params_dicts[param_name]
            # '/f_6_.W' --> 'f_6_.W'
            slash_index = param_name.find('/')
            param_name = param_name[slash_index + 1:]
            assert param.get_value().shape == loaded[param_name].shape
            param.set_value(loaded[param_name])

    test_data_stream, test_data_stream = get_mixed_streams(10000)
    test_data = test_data_stream.get_epoch_iterator().next()
    test_data_input = test_data[10]
    test_data_target = test_data[0]
    print 'Compiling ...'
    cg = ComputationGraph([ladder.costs.total])
    eval_ = theano.function(cg.inputs, ladder.error)
    print 'Test_set_Error: ' + str(eval_(test_data_input, test_data_target))
    import ipdb
    ipdb.set_trace()

    # from ploting2 import bar_chart, plot_representations, compute_noises
    # compute_noises(ladder)
    # plot_representations(ladder, params_dicts)
    # bar_chart(params_dicts)


if __name__ == "__main__":
    # load_path = '/u/pezeshki/ladder_network/results/mnist_standard_2015_10_08_at_15_53'
    load_path = None
    logging.basicConfig(level=logging.INFO)
    ladder = setup_model()
    if load_path is None:
        train(ladder)
    else:
        evaluate(ladder, load_path)
