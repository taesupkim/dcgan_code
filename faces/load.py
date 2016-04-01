import sys
sys.path.append('..')

import os
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream

from lib.config import data_dir

FACE_DATAPATH = '/data/lisatmp4/taesup/data/face/CelebA/CelebFace_64x64.hdf5'
def faces(batch_size=128):
    tr_data = H5PYDataset(FACE_DATAPATH, which_sets=('train',))

    tr_scheme = ShuffledScheme(examples=tr_data.num_examples, batch_size=batch_size)
    tr_stream = DataStream(tr_data, iteration_scheme=tr_scheme)

    return tr_data,tr_stream