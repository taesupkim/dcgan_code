from numpy.random import RandomState
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from random import Random

seed = 42

py_rng = Random(seed)
np_rng = RandomState(seed)
t_rng = RandomStreams(seed)