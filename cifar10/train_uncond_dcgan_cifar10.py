import sys
sys.path.append('..')
import os
import json
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch

from load import cifar10

def transform(X):
    return floatX(X)/127.5 - 1.

def inverse_transform(X):
    X = (X+1.)/2.
    return X

k = 1             # # of discrim updates for each gen update
l2 = 1e-5         # l2 weight decay
nvis = 196        # # of samples to visualize during training
b1 = 0.5          # momentum term of adam
nc = 3            # # of channels in image
nbatch = 128      # # of examples in batch
npx = 16          # # of pixels width/height of images

nx = npx*npx*nc   # # of dimensions in X
niter = 100        # # of iter at starting learning rate
niter_decay = 0   # # of iter to linearly decay learning rate to zero
lr = 0.001       # initial learning rate for adam
ntrain = 50000   # # of examples to train on

###################
# SET OUTPUT PATH #
###################
desc = 'dcgan_cifar10_16x16'
model_dir = 'models/%s'%desc
samples_dir = 'samples/%s'%desc
if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

##############################
# SET ACTIVATIONS AND OTHERS #
##############################
relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy

###################
# SET INITIALIZER #
###################
gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)

#################
# LOAD DATA SET #
#################
tr_data, te_data, tr_stream, val_stream, te_stream = cifar10(ntrain=ntrain, window_size=(npx, npx))

###################
# GET DATA STATIC #
###################
tr_handle = tr_data.open()
vaX, = tr_data.get_data(tr_handle, slice(0, 10000))
vaX = transform(vaX)

#####################
# INITIALIZE PARAMS #
#####################
nz  = 500 # NUM OF HIDDENS
ngf = ndf = 128  # NUM OF MINIMAL FILTERS
# FOR GENERATOR
#   LAYER 1 (LINEAR)
gw  = gifn((nz, ngf*4*(2*2)), 'gw')
gg = gain_ifn((ngf*4*(2*2)), 'gg')
gb = bias_ifn((ngf*4*(2*2)), 'gb')
#   LAYER 2 (DECONV)
gw2 = gifn((ngf*4, ngf*2, 5, 5), 'gw2')
gg2 = gain_ifn((ngf*2), 'gg2')
gb2 = bias_ifn((ngf*2), 'gb2')
#   LAYER 3 (DECONV)
gw3 = gifn((ngf*2, ngf*1, 5, 5), 'gw3')
gg3 = gain_ifn((ngf*1), 'gg3')
gb3 = bias_ifn((ngf*1), 'gb3')
#   LAYER 4 (DECONV)
gwx = gifn((ngf*1, nc, 5, 5), 'gwx')

# FOR DISCRIMINATOR
#   LAYER 0 (DECONV)
dw  = difn((ndf, nc, 5, 5), 'dw')
#   LAYER 1 (DECONV)
dw2 = difn((ndf*2, ndf, 5, 5), 'dw2')
dg2 = gain_ifn((ndf*2), 'dg2')
db2 = bias_ifn((ndf*2), 'db2')
#   LAYER 2 (DECONV)
dw3 = difn((ndf*4, ndf*2, 5, 5), 'dw3')
dg3 = gain_ifn((ndf*4), 'dg3')
db3 = bias_ifn((ndf*4), 'db3')
#   LAYER 4 (LINEAR)
dwy = difn((ndf*4*(2*2), 1), 'dwy')
# SET AS LIST
gen_params = [gw, gg, gb, gw2, gg2, gb2, gw3, gg3, gb3, gwx]
discrim_params = [dw, dw2, dg2, db2, dw3, dg3, db3, dwy]

###################
# BUILD GENERATOR #
###################
def gen(Z, w, g, b, w2, g2, b2, w3, g3, b3,wx):
    h = relu(batchnorm(T.dot(Z, w), g=g, b=b))
    h = h.reshape((h.shape[0], ngf*4, 2, 2))
    h2 = relu(batchnorm(deconv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    x = tanh(deconv(h3, wx, subsample=(2, 2), border_mode=(2, 2)))
    return x

#######################
# BUILD DISCRIMINATOR #
#######################
def discrim(X, w, w2, g2, b2, w3, g3, b3, wy):
    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h3 = T.flatten(h3, 2)
    y = sigmoid(T.dot(h3, wy))
    return y
##################
# SET INPUT DATA #
##################
X = T.tensor4()
Z = T.matrix()

###################
# GENERATE SAMPLE #
###################
gX = gen(Z, *gen_params)

###########################
# GET DISCRIMINATOR SCORE #
###########################
p_real = discrim(X, *discrim_params)
p_gen = discrim(gX, *discrim_params)

######################################
# SET DISCRIMINATOR & GENERATOR COST #
######################################
d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()

d_cost = d_cost_real + d_cost_gen
g_cost = g_cost_d

cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

###############
# SET UPDATER #
###############
lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
updates = d_updates + g_updates

######################################
# RANDOM SELECT INPUT DATA & DISPLAY #
######################################
vis_idxs = py_rng.sample(np.arange(len(vaX)), nvis)
vaX_vis = inverse_transform(vaX[vis_idxs])
color_grid_vis(vaX_vis.transpose([0,2,3,1]), (14, 14), 'samples/%s_etl_test.png'%desc)


####################
# COMPILE FUNCTION #
####################
print 'COMPILING'
t = time()
_train_g = theano.function([X, Z], cost, updates=g_updates)
_train_d = theano.function([X, Z], cost, updates=d_updates)
_gen = theano.function([Z], gX)
print '%.2f seconds to compile theano functions'%(time()-t)


#####################################
# SAMPLE RANDOM DATA FOR GENERATION #
#####################################
sample_zmb = floatX(np_rng.uniform(-1., 1., size=(nvis, nz)))

###################
# GENERATE SAMPLE #
###################
def gen_samples(n, nbatch=128):
    samples = []
    n_gen = 0
    for i in range(n/nbatch):
        zmb = floatX(np_rng.uniform(-1., 1., size=(nbatch, nz)))
        xmb = _gen(zmb)
        samples.append(xmb)
        n_gen += len(xmb)
    n_left = n-n_gen
    zmb = floatX(np_rng.uniform(-1., 1., size=(n_left, nz)))
    xmb = _gen(zmb)
    samples.append(xmb)
    return np.concatenate(samples, axis=0)

f_log = open('logs/%s.ndjson'%desc, 'wb')
log_fields = [
    'n_epochs',
    'n_updates',
    'n_examples',
    'n_seconds',
    '1k_va_nnd',
    '10k_va_nnd',
    '100k_va_nnd',
    'g_cost',
    'd_cost',
]

vaX = vaX.reshape(len(vaX), -1)


##################
# START TRAINING #
##################
print desc.upper()
n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()

# FOR EACH EPOCH
for epoch in range(niter):
    # FOR EACH BATCH
    print "CURRENT EPOCH .{}".format(epoch+1)
    for imb, in tqdm(tr_stream.get_epoch_iterator(), total=ntrain/nbatch):
        # GET NORMALIZED INPUT DATA
        imb = transform(imb)
        # GET INPUT RANDOM DATA FOR SAMPLING
        zmb = floatX(np_rng.uniform(-1., 1., size=(len(imb), nz)))
        # UPDATE MODEL
        if n_updates % (k+1) == 0:
            cost = _train_g(imb, zmb)
        else:
            cost = _train_d(imb, zmb)
        n_updates += 1
        n_examples += len(imb)
    # GET RECENT RESULT
    g_cost = float(cost[0])
    d_cost = float(cost[1])

    # # GENERATE SAMPLE
    # gX = gen_samples(100000)
    # gX = gX.reshape(len(gX), -1)
    # va_nnd_1k = nnd_score(gX[:1000], vaX, metric='euclidean')
    # va_nnd_10k = nnd_score(gX[:10000], vaX, metric='euclidean')
    # va_nnd_100k = nnd_score(gX[:100000], vaX, metric='euclidean')
    # log = [n_epochs, n_updates, n_examples, time()-t, va_nnd_1k, va_nnd_10k, va_nnd_100k, g_cost, d_cost]
    # print '%.0f %.2f %.2f %.2f %.4f %.4f'%(epoch, va_nnd_1k, va_nnd_10k, va_nnd_100k, g_cost, d_cost)
    # f_log.write(json.dumps(dict(zip(log_fields, log)))+'\n')
    # f_log.flush()

    # KEEP SAMPLE FROM SAME NOISE INPUT
    samples = np.asarray(_gen(sample_zmb))
    color_grid_vis(inverse_transform(samples).transpose([0,2,3,1]), (14, 14), 'samples/%s/%d.png'%(desc, n_epochs))
    n_epochs += 1
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
    # if n_epochs in [1, 2, 3, 4, 5, 10, 15, 20, 25]:
    #     joblib.dump([p.get_value() for p in gen_params], 'models/%s/%d_gen_params.jl'%(desc, n_epochs))
    #     joblib.dump([p.get_value() for p in discrim_params], 'models/%s/%d_discrim_params.jl'%(desc, n_epochs))