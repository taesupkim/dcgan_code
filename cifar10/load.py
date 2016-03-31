import sys
sys.path.append('..')

import os
from fuel.datasets.cifar10 import CIFAR10
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers.image import RandomFixedSizeCrop
from fuel.datasets.imagenet import ILSVRC2010

def cifar10(ntrain=None,  ntest=None, window_size=(32, 32), batch_size=128):

    tr_data = CIFAR10(which_sets=('train',), sources=('features',))
    te_data = CIFAR10(which_sets=('test',), sources=('features',))

    if ntrain is None:
        ntrain = tr_data.num_examples
    if ntest is None:
        ntest = te_data.num_examples

    tr_scheme = ShuffledScheme(examples=ntrain, batch_size=batch_size)
    tr_stream = DataStream(tr_data, iteration_scheme=tr_scheme)
    tr_stream = RandomFixedSizeCrop(tr_stream, window_size)

    te_scheme = SequentialScheme(examples=ntest, batch_size=batch_size)
    te_stream = DataStream(te_data, iteration_scheme=te_scheme)
    te_stream = RandomFixedSizeCrop(te_stream, window_size)

    return tr_data, te_data, tr_stream, te_stream