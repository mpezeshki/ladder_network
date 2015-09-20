import logging
import numpy
from fuel.streams import DataStream
from fuel.transformers import Transformer
from picklable_itertools import cycle, imap

from timit import (Timit, WindowFeatures, Reshape, SequentialShuffledScheme)

logger = logging.getLogger('datasets')


def get_mixed_datastream(dataset, indices, batch_size,
                         n_labeled, n_unlabeled):

    # Ensure each label is equally represented
    logger.info('Balancing %d labels...' % n_labeled)
    all_data = dataset.data_sources[1]
    y = all_data.flatten()[indices]
    n_classes = y.max()  # indices of classes starts from 1
    # assert n_labeled % n_classes == 0
    n_from_each_class = n_labeled / n_classes

    i_labeled = []
    for c in range(n_classes):
        i = (indices[y == c])[:n_from_each_class]
        i_labeled += list(i)

    # Get unlabeled indices
    i_unlabeled = indices[:n_unlabeled]

    data_stream = DataStream.default_stream(
        dataset,
        iteration_scheme=SequentialShuffledScheme(dataset.num_examples,
                                                  batch_size=1))

    reshaped_stream = Reshape('features',
                              'features_shapes',
                              data_stream=data_stream)
    window_stream = WindowFeatures(reshaped_stream, 'features', 13)

    ds = CombinedDataStream(
        data_stream_labeled=FlattenTimit(window_stream),
        data_stream_unlabeled=FlattenTimit(window_stream))

    return ds


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


class FlattenTimit(Transformer):
    def __init__(self, data_stream, **kwargs):
        super(FlattenTimit, self).__init__(data_stream,
                                           **kwargs)

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        flatten_data = []
        flatten_data.append(data[0][0].reshape((data[0][0].shape[0], -1)))
        flatten_data.append(data[1][0].reshape((data[1][0].shape[0], )))
        return flatten_data


def get_dataset(unlabeled_samples, valid_set_size, test_set=False):
    dataset = Timit("train")

    # Take all indices and permutate them
    all_ind = numpy.arange(dataset.num_examples)
    rng = numpy.random.RandomState(seed=1)
    rng.shuffle(all_ind)

    indices = []
    # Train
    indices.append(all_ind[:unlabeled_samples])
    # Validation
    indices.append(numpy.setdiff1d(all_ind, indices[0])[:valid_set_size])

    return dataset, indices
