import sys
sys.path.append('..')

import os
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream

from lib.config import data_dir

LSUN_BEDROOM_PATH = '/data/lisatmp4/taesup/data/lsun/lsun_bedroom_train_64x64.hdf5',
def bedroom(batch_size=128):
    print LSUN_BEDROOM_PATH
    tr_data = H5PYDataset(LSUN_BEDROOM_PATH, which_sets=('train',))

    tr_scheme = ShuffledScheme(examples=tr_data.num_examples, batch_size=batch_size)
    tr_stream = DataStream(tr_data, iteration_scheme=tr_scheme)

    return tr_data, tr_stream