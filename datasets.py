import numpy
from fuel.datasets import MNIST
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer


class Flatten(Transformer):
    def __init__(self, data_stream, **kwargs):
        super(Flatten, self).__init__(data_stream,
                                      **kwargs)

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        flatten_data = []
        flatten_data.append(data[0].reshape((data[0].shape[0], -1)))
        flatten_data.append(data[1].reshape((data[1].shape[0], )))
        return flatten_data


def get_streams(num_train_examples, batch_size, use_test=True):
    dataset = MNIST(("train",))
    all_ind = numpy.arange(dataset.num_examples)
    rng = numpy.random.RandomState(seed=1)
    rng.shuffle(all_ind)

    indices_train = all_ind[:num_train_examples]
    indices_valid = all_ind[num_train_examples:]

    tarin_stream = Flatten(DataStream.default_stream(
        dataset,
        iteration_scheme=ShuffledScheme(indices_train, batch_size)))

    valid_stream = None
    if len(indices_valid) != 0:
        valid_stream = Flatten(DataStream.default_stream(
            dataset,
            iteration_scheme=ShuffledScheme(indices_valid, batch_size)))

    test_stream = None
    if use_test:
        dataset = MNIST(("test",))
        ind = numpy.arange(dataset.num_examples)
        rng = numpy.random.RandomState(seed=1)
        rng.shuffle(all_ind)

        test_stream = Flatten(DataStream.default_stream(
            dataset,
            iteration_scheme=ShuffledScheme(ind, batch_size)))

    return tarin_stream, valid_stream, test_stream
