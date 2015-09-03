import logging
import numpy
from fuel.datasets import MNIST
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer
from picklable_itertools import cycle, imap
from utils import AttributeDict

logger = logging.getLogger('datasets')


def make_datastream(dataset, indices, batch_size,
                    n_labeled=None, n_unlabeled=None,
                    balanced_classes=True, whiten=None, cnorm=None,
                    scheme=ShuffledScheme):

    # Ensure each label is equally represented
    logger.info('Balancing %d labels...' % n_labeled)
    all_data = dataset.data_sources[dataset.sources.index('targets')]
    y = all_data.flatten()[indices]
    n_classes = y.max() + 1
    assert n_labeled % n_classes == 0
    n_from_each_class = n_labeled / n_classes

    i_labeled = []
    for c in range(n_classes):
        i = (indices[y == c])[:n_from_each_class]
        i_labeled += list(i)

    # Get unlabeled indices
    i_unlabeled = indices[:n_unlabeled]

    ds = CombinedDataStream(
        data_stream_labeled=MyTransformer(
            DataStream(dataset),
            iteration_scheme=scheme(i_labeled, batch_size)),
        data_stream_unlabeled=MyTransformer(
            DataStream(dataset),
            iteration_scheme=scheme(i_unlabeled, batch_size))
    )
    return ds


class MyTransformer(Transformer):
    def __init__(self, data_stream, iteration_scheme, **kwargs):
        super(MyTransformer, self).__init__(data_stream,
                                            iteration_scheme=iteration_scheme,
                                            **kwargs)
        data = data_stream.get_data(slice(data_stream.dataset.num_examples))
        shape = data[0].shape
        self.data = [data[0].reshape(shape[0], -1)]
        self.data += [data[1].flatten()]

    def get_data(self, request=None):
        return (s[request] for s in self.data)


class CombinedDataStream(Transformer):
    def __init__(self, data_stream_labeled, data_stream_unlabeled, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.ds_labeled = data_stream_labeled
        self.ds_unlabeled = data_stream_unlabeled
        # Rename the sources for clarity
        self.ds_labeled.sources = ('features_labeled', 'targets_labeled')
        # Hide the labels.
        self.ds_unlabeled.sources = ('features_unlabeled',)

    @property
    def sources(self):
        if hasattr(self, '_sources'):
            return self._sources
        return self.ds_labeled.sources + self.ds_unlabeled.sources

    @sources.setter
    def sources(self, value):
        self._sources = value

    def close(self):
        self.ds_labeled.close()
        self.ds_unlabeled.close()

    def reset(self):
        self.ds_labeled.reset()
        self.ds_unlabeled.reset()

    def next_epoch(self):
        self.ds_labeled.next_epoch()
        self.ds_unlabeled.next_epoch()

    def get_epoch_iterator(self, **kwargs):
        unlabeled = self.ds_unlabeled.get_epoch_iterator(**kwargs)
        labeled = self.ds_labeled.get_epoch_iterator(**kwargs)
        assert type(labeled) == type(unlabeled)

        return imap(self.mergedicts, cycle(labeled), unlabeled)

    def mergedicts(self, x, y):
        return dict(list(x.items()) + list(y.items()))


def get_mnist_data_dict(unlabeled_samples=60000, valid_set_size=10000,
                        test_set=False):
    train_set = MNIST(("train",))
    # Make sure the MNIST data is in right format
    train_set.data_sources = (
        (train_set.data_sources[0] / 255.).astype(numpy.float32),
        train_set.data_sources[1])

    # Take all indices and permutate them
    all_ind = numpy.arange(train_set.num_examples)
    rng = numpy.random.RandomState(seed=1)
    rng.shuffle(all_ind)

    data = AttributeDict()

    # Choose the training set
    data.train = train_set
    data.train_ind = all_ind[:unlabeled_samples]

    # Then choose validation set from the remaining indices
    data.valid = train_set
    data.valid_ind = numpy.setdiff1d(all_ind, data.train_ind)[:valid_set_size]
    logger.info('Using %d examples for validation' % len(data.valid_ind))

    # Only touch test data if requested
    if test_set:
        data.test = MNIST(("test",))
        data.test_ind = numpy.arange(data.test.num_examples)

    return data
