import logging
import numpy
from fuel.transformers import Transformer
from collections import OrderedDict
from picklable_itertools.extras import equizip
import tables
from fuel.datasets.hdf5 import PytablesDataset
from fuel import config

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

        # For compatibility with MNIST
        self.data_sources = []
        self.data_sources.append(numpy.array([]))
        self.data_sources.append(numpy.hstack(
            [x for x in self.nodes[2].iterrows()]))


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
        self.dataset = self.data_stream.dataset

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        data = OrderedDict(zip(self.data_stream.sources, data))
        shapes = data.pop(self.shape_source)
        reshaped_data = []
        for dt, shape in zip(data[self.data_source], shapes):
            reshaped_data.append(dt.reshape(shape))
        data[self.data_source] = reshaped_data
        return data.values()


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
    def __init__(self, num_examples, batch_size):
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.rng = numpy.random.RandomState(config.default_seed)

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
