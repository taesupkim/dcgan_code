import sys
sys.path.append('..')

import os
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream

from lib.config import data_dir

IMAGENET_DATAPATH = '/data/lisatmp4/taesup/data/imagenet64x64/imagenet64x64_train.hdf5'
def imagenet(batch_size=128):
    tr_data = H5PYDataset(IMAGENET_DATAPATH, which_sets=('train',))

    tr_scheme = ShuffledScheme(examples=tr_data.num_examples, batch_size=batch_size)
    tr_stream = DataStream(tr_data, iteration_scheme=tr_scheme)

    return tr_data, tr_stream