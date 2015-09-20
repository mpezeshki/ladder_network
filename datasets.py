import logging
import numpy
from fuel.datasets import MNIST
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer
from picklable_itertools import cycle, imap
from utils import AttributeDict
from collections import OrderedDict
from picklable_itertools.extras import equizip
import tables
from fuel.datasets.hdf5 import PytablesDataset

logger = logging.getLogger('datasets')


class Timit(PytablesDataset):
    def __init__(self, which_set):
        self.path = '/data/lisatmp3/speech/timit_fbank_framewise.h5'
        self.which_set = which_set
        self.sources = ('features', 'features_shapes', 'phonemes')
        super(Timit, self).__init__(
            self.path, self.sources, data_node=which_set)

    def open_file(self, path):
        # CAUTION: This is a hack!
        # Use `open_file` when Fred updates os
        self.h5file = tables.File(path, mode="r")
        node = self.h5file.get_node('/', self.data_node)

        self.nodes = [getattr(node, source) for source in self.sources_in_file]
        if self.stop is None:
            self.stop = self.nodes[0].nrows
        self.num_examples = self.stop - self.start


class WindowFeatures(Transformer):
    def __init__(self, data_stream, source, window_size):
        super(WindowFeatures, self).__init__(data_stream)
        self.source = source
        self.window_size = window_size

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        data = OrderedDict(equizip(self.sources, data))
        feature_batch = data[self.source]

        windowed_features = []
        for features in feature_batch:
            features_padded = features.copy()

            features_shifted = [features]
            # shift forward
            for i in xrange(self.window_size / 2):
                feats = numpy.roll(features_padded, i + 1, axis=0)
                feats[:i + 1, :] = 0
                features_shifted.append(feats)
            features_padded = features.copy()

            # shift backward
            for i in xrange(self.window_size / 2):
                feats = numpy.roll(features_padded, -i - 1, axis=0)
                feats[-i - 1:, :] = 0
                features_shifted.append(numpy.roll(features_padded, -i - 1,
                                                   axis=0))
            windowed_features.append(numpy.concatenate(
                features_shifted, axis=1))
        data[self.source] = windowed_features
        return data.values()


class Reshape(Transformer):
    """Reshapes data in the stream according to shape source."""
    def __init__(self, data_source, shape_source, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.data_source = data_source
        self.shape_source = shape_source
        self.sources = tuple(source for source in self.data_stream.sources
                             if source != shape_source)

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        data = OrderedDict(zip(self.data_stream.sources, data))
        shapes = data.pop(self.shape_source)
        reshaped_data = []
        for dt, shape in zip(data[self.data_source], shapes):
            reshaped_data.append(dt.reshape(shape))
        data[self.data_source] = reshaped_data
        return data.values()


def make_datastream(dataset, indices, batch_size,
                    n_labeled=None, n_unlabeled=None,
                    scheme=ShuffledScheme, is_timit=False):

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

    if is_timit:
        data_stream = DataStream(t, iteration_scheme=SequentialShuffledScheme(
            t.num_examples, t.num_examples, rng))

        reshaped_stream = Reshape('features', 'features_shapes',
                                  data_stream=data_stream)

        window_stream = WindowFeatures(reshaped_stream, 'features', 13)

        ds = CombinedDataStream(
            data_stream_labeled=MyTransformer2(
                window_stream,
                iteration_scheme=scheme(i_labeled, batch_size)),
            data_stream_unlabeled=MyTransformer2(
                window_stream,
                iteration_scheme=scheme(i_unlabeled, batch_size)))
    else:
        ds = CombinedDataStream(
            data_stream_labeled=MyTransformer(
                DataStream(dataset),
                iteration_scheme=scheme(i_labeled, batch_size)),
            data_stream_unlabeled=MyTransformer(
                DataStream(dataset),
                iteration_scheme=scheme(i_unlabeled, batch_size)))

    return ds


class MyTransformer(Transformer):
    def __init__(self, data_stream, iteration_scheme, **kwargs):
        super(MyTransformer, self).__init__(data_stream,
                                            iteration_scheme=iteration_scheme,
                                            **kwargs)
        # import ipdb; ipdb.set_trace()
        data = data_stream.get_data(slice(data_stream.dataset.num_examples))
        shape = data[0].shape
        self.data = [data[0].reshape(shape[0], -1)]
        self.data += [data[1].flatten()]

    def get_data(self, request=None):
        return (s[request] for s in self.data)


class MyTransformer2(Transformer):
    def __init__(self, data_stream, iteration_scheme, **kwargs):
        super(MyTransformer2, self).__init__(data_stream,
                                             iteration_scheme=iteration_scheme,
                                             **kwargs)
        it = data_stream.get_epoch_iterator()
        data = it.next()
        utterances, phonemes_seq = data
        utterances = numpy.vstack(utterances)
        phonemes_seq = numpy.vstack(phonemes_seq)

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


def get_mnist_data_dict(unlabeled_samples, valid_set_size, test_set=False):
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


def get_timit_data_dict(unlabeled_samples, valid_set_size, test_set=False):
    train_set = Timit('train')
    all_ind = numpy.arange(1124823)
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

    return data


import six
import math
from fuel.schemes import BatchScheme


class SequentialShuffledScheme(BatchScheme):
    """Sequential batches iterator.
    Iterate over all the examples in a dataset of fixed size sequentially
    in batches of a given size.
    Notes
    -----
    The batch size isn't enforced, so the last batch could be smaller.
    """
    def __init__(self, num_examples, batch_size, rng):
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.rng = rng

    def get_request_iterator(self):
        return SequentialShuffledIterator(self.num_examples, self.batch_size,
                                          self.rng)


class SequentialShuffledIterator(six.Iterator):
    def __init__(self, num_examples, batch_size, rng):
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.rng = rng
        self.batch_indexes = range(int(math.ceil(
            num_examples / float(batch_size))))
        self.rng.shuffle(self.batch_indexes)
        self.current = 0
        self.current_batch = 0

    def __iter__(self):
        self.rng.shuffle(self.batch_indexes)
        return self

    def __next__(self):
        if self.current >= self.num_examples:
            raise StopIteration
        current_index = self.batch_indexes[self.current_batch]
        slice_ = slice(current_index * self.batch_size,
                       min(self.num_examples,
                           (current_index + 1) * self.batch_size))
        self.current += self.batch_size
        self.current_batch += 1
        return slice_
