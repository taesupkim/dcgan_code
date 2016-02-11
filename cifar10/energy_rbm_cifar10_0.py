import os
import numpy as np
from time import time
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib.activations import Rectify, Tanh, Softplus
from lib.updates import Adagrad, Regularizer
from lib.inits import Normal, Constant
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.theano_utils import floatX, sharedX

from load import cifar10

def transform(X):
    return floatX(X)/127.5 - 1.

def inverse_transform(X):
    X = (X+1.)/2.
    return X
def plot_learning_curve(cost_values, cost_names, save_as):
    import matplotlib.pyplot as plt
    for cost in cost_values:
        print cost

        plt.plot(xrange(len(cost)), cost)

    plt.legend(cost_names, loc='upper right')
    plt.savefig(save_as)
    plt.close()

model_name  = 'ENERGY_RBM_CIFAR10'
samples_dir = 'samples/%s'%model_name
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)


num_channels = 3
input_shape  = 32
input_size   = input_shape*input_shape*num_channels
num_display  = 16*16
batch_size   = 128
constant     = 1.0

##############################
# SET ACTIVATIONS AND OTHERS #
##############################
relu = Rectify()
tanh = Tanh()
softplus = Softplus()

###################
# SET INITIALIZER #
###################
gifn = Normal(scale=0.01)
difn = Normal(scale=0.01)
gain_ifn = Normal(loc=1., scale=0.02)
bias_ifn = Constant(c=0.)

#################
# LOAD DATA SET #
#################
train_data, test_data, train_stream, valid_stream, test_stream = cifar10(window_size=(input_shape, input_shape), batch_size=batch_size)

#####################
# INITIALIZE PARAMS #
#####################
filter_size  = 5
num_hiddens  = 100 # NUM OF HIDDENS
num_layers   = 4
min_num_gen_filters = min_num_eng_filters = 128

###################
# BUILD GENERATOR #
###################
init_image_size  = 4
num_gen_filters0 = min_num_gen_filters*4
num_gen_filters1 = min_num_gen_filters*2
num_gen_filters2 = min_num_gen_filters
# LAYER 0 (LINEAR)
linear_w0 = gifn((num_hiddens, num_gen_filters0*init_image_size*init_image_size), 'linear_w0')
bn_w0     = gain_ifn((num_gen_filters0*init_image_size*init_image_size), 'bn_w0')
bn_b0     = bias_ifn((num_gen_filters0*init_image_size*init_image_size), 'bn_b0')
# LAYER 1 (DECONV)
conv_w1   = gifn((num_gen_filters0, num_gen_filters1, filter_size, filter_size), 'conv_w1')
bn_w1     = gain_ifn(num_gen_filters1, 'bn_w1')
bn_b1     = bias_ifn(num_gen_filters1, 'bn_b1')
# LAYER 2 (DECONV)
conv_w2   = gifn((num_gen_filters1, num_gen_filters2, filter_size, filter_size), 'conv_w2')
bn_w2     = gain_ifn(num_gen_filters2, 'bn_w2')
bn_b2     = bias_ifn(num_gen_filters2, 'bn_b2')
# LAYER 3 (DECONV)
conv_w3   = gifn((num_gen_filters2, num_channels, filter_size, filter_size), 'conv_w3')
# PARAM SET
generator_params = [linear_w0, bn_w0, bn_b0, conv_w1, bn_w1, bn_b1, conv_w2, bn_w2, bn_b2, conv_w3]
def generator_model(hidden_data,
                    linear_w0,
                    bn_w0,
                    bn_b0,
                    conv_w1,
                    bn_w1,
                    bn_b1,
                    conv_w2,
                    bn_w2,
                    bn_b2,
                    conv_w3,
                    is_training=True):
    h0     = relu(batchnorm(X=T.dot(hidden_data, linear_w0), g=bn_w0, b=bn_b0))
    h0     = h0.reshape((h0.shape[0], num_gen_filters0, init_image_size, init_image_size))
    h1     = relu(batchnorm(deconv(h0, conv_w1, subsample=(2, 2), border_mode=(2, 2)), g=bn_w1, b=bn_b1))
    h2     = relu(batchnorm(deconv(h1, conv_w2, subsample=(2, 2), border_mode=(2, 2)), g=bn_w2, b=bn_b2))
    output = tanh(deconv(h2, conv_w3, subsample=(2, 2), border_mode=(2, 2)))
    return output

######################
# BUILD ENERGY MODEL #
######################
min_image_size   = init_image_size
num_eng_filters0 = min_num_eng_filters
num_eng_filters1 = min_num_eng_filters*2
num_eng_filters2 = min_num_eng_filters*4
# LAYER 0 (DECONV)
conv_w0   = difn((num_eng_filters0, num_channels, filter_size, filter_size), 'conv_w0')
#   LAYER 1 (DECONV)
conv_w1   = difn((num_eng_filters1, num_eng_filters0, filter_size, filter_size), 'conv_w1')
bn_w1     = gain_ifn(num_eng_filters1, 'bn_w1')
bn_b1     = bias_ifn(num_eng_filters1, 'bn_b1')
#   LAYER 2 (DECONV)
conv_w2   = difn((num_eng_filters2, num_eng_filters1, filter_size, filter_size), 'conv_w2')
bn_w2     = gain_ifn(num_eng_filters2, 'bn_w2')
bn_b2     = bias_ifn(num_eng_filters2, 'bn_b2')
#   LAYER 3 (LINEAR)
linear_w3 = difn((num_eng_filters2*(min_image_size*min_image_size), num_hiddens), 'linear_w3')
linear_b3 = bias_ifn(num_hiddens, 'linear_b3')
# SET AS LIST
energy_params = [conv_w0, conv_w1, bn_w1, bn_b1, conv_w2, bn_w2, bn_b2, linear_w3, linear_b3]
def energy_model(input_data,
                 conv_w0,
                 conv_w1,
                 bn_w1,
                 bn_b1,
                 conv_w2,
                 bn_w2,
                 bn_b2,
                 linear_w3,
                 linear_b3,
                 is_training=True):
    h0 = relu(dnn_conv(input_data, conv_w0, subsample=(2, 2), border_mode=(2, 2)))
    h1 = dropout(relu(batchnorm(dnn_conv(h0, conv_w1, subsample=(2, 2), border_mode=(2, 2)), g=bn_w1, b=bn_b1)), p=0.5, is_training=is_training)
    h2 = dropout(tanh(batchnorm(dnn_conv(h1, conv_w2, subsample=(2, 2), border_mode=(2, 2)), g=bn_w2, b=bn_b2)), p=0.5, is_training=is_training)
    h2 = T.flatten(h2, 2)
    y  = softplus(T.dot(h2, linear_w3)+linear_b3)
    y  = T.sum(-y, axis=1)
    return y

def set_update_function(energy_params, generator_params, energy_updater, generator_updater):
    # set input data, hidden data, noise data
    input_data  = T.tensor4(name='input_data', dtype=theano.config.floatX)
    hidden_data = T.matrix(name='hidden_data', dtype=theano.config.floatX)
    noise_data  = T.tensor4(name='noise_data', dtype=theano.config.floatX)

    # get sample data
    sample_data = generator_model(hidden_data, *generator_params, is_training=True)
    sample_data = T.clip(sample_data+noise_data, -1.0, 1.0)

    # put together, get energy
    whole_data   = T.concatenate([input_data, sample_data], axis=0)
    whole_energy = energy_model(whole_data, *energy_params, is_training=True)

    input_energy  = whole_energy[:input_data.shape[0]]
    sample_energy = whole_energy[input_data.shape[0]:]

    # set generator cost
    generator_cost    = sample_energy.mean()
    generator_updates = generator_updater(generator_params, generator_cost)

    # set energy function cost
    importance_rate = T.nnet.softmax(T.transpose(-sample_energy))
    importance_rate = theano.gradient.disconnected_grad(importance_rate)

    energy_cost    = input_energy.mean()-T.dot(importance_rate, sample_energy).sum()
    energy_updates = generator_updater(energy_params, energy_cost)

    function_inputs  = [input_data, hidden_data, noise_data]
    function_outputs = [input_energy, sample_energy]

    function = theano.function(inputs=function_inputs,
                               outputs=function_outputs,
                               updates=generator_updates+energy_updates,
                               on_unused_input='ignore')
    return function

def set_evaluation_and_sampling_function(energy_params, generator_params):
    # input data
    input_data  = T.tensor4(name='input_data', dtype=theano.config.floatX)
    hidden_data = T.matrix(name='hidden_data', dtype=theano.config.floatX)

    # get sample data
    sample_data = generator_model(hidden_data, *generator_params, is_training=False)

    # put together, get energy
    whole_data   = T.concatenate([input_data, sample_data], axis=0)
    whole_energy = energy_model(whole_data, *energy_params, is_training=False)

    input_energy  = whole_energy[:input_data.shape[0]]
    sample_energy = whole_energy[input_data.shape[0]:]

    function_inputs  = [input_data, hidden_data]
    function_outputs = [input_energy, sample_energy, sample_data]

    function = theano.function(inputs=function_inputs,
                               outputs=function_outputs,
                               on_unused_input='ignore')
    return function

def set_sampling_function(generator_params):
    # hidden data
    hidden_data = T.matrix(name='hidden_data', dtype=theano.config.floatX)

    # get sample data
    sample_data = generator_model(hidden_data, *generator_params, is_training=False)

    sampling_function_inputs  = [hidden_data]
    sampling_function_outputs = [sample_data]

    sampling_function = theano.function(inputs=sampling_function_inputs,
                                        outputs=sampling_function_outputs,
                                        on_unused_input='ignore')
    return sampling_function

def train_model(learning_rate=1e-2,
                lambda_eng=1e-5,
                lambda_gen=1e-5,
                init_noise=0.1,
                noise_decay=1.0,
                batch_size=100,
                num_epochs=100):
    model_test_name = model_name \
                      + '_LAYERS{}'.format(int(num_layers)) \
                      + '_HIDDEN{}'.format(int(num_hiddens)) \
                      + '_LR{}'.format(int(-np.log10(learning_rate))) \
                      + '_L(ENG){}'.format(int(-np.log10(lambda_eng))) \
                      + '_L(GEN){}'.format(int(-np.log10(lambda_gen))) \
                      + '_NOISE{0:.2f}'.format(float(init_noise)) \
                      + '_DECAY{0:.2f}'.format(float(noise_decay)) \
    # set updates
    energy_updater    = Adagrad(lr=sharedX(learning_rate), regularizer=Regularizer(l2=lambda_eng))
    generator_updater = Adagrad(lr=sharedX(learning_rate), regularizer=Regularizer(l2=lambda_gen))

    # compile function
    print 'COMPILING'
    t=time()
    update_function      = set_update_function(energy_params, generator_params, energy_updater, generator_updater)
    eval_sample_function = set_evaluation_and_sampling_function(energy_params, generator_params)
    sampler_function     = set_sampling_function(generator_params)
    print '%.2f seconds to compile theano functions'%(time()-t)

    # set fixed hidden data (for sampling
    fixed_hidden_data = floatX(np_rng.uniform(-constant, constant, size=(num_display, num_hiddens)))

    # tracking variable
    # train output
    train_input_energy  = []
    train_sample_energy = []
    # valid output
    valid_input_energy  = []
    valid_sample_energy = []

    # for each epoch
    for e in xrange(num_epochs):
        # train phase
        epoch_train_input_energy  = 0.
        epoch_train_sample_energy = 0.
        epoch_train_count         = 0.

        train_batch_iters = train_stream.get_epoch_iterator()
        # for each batch
        for b, train_batch_data in enumerate(train_batch_iters):
            # set update function inputs
            input_data   = transform(train_batch_data[0])
            hidden_data  = floatX(np_rng.uniform(low=-constant, high=constant, size=(input_data.shape[0], num_hiddens)))
            sample_noise = floatX(np_rng.normal(size=input_data.shape)*init_noise*(noise_decay**e))
            # update function
            [input_energy, sample_energy] = update_function(input_data, hidden_data, sample_noise)

            # get output values
            epoch_train_input_energy  += input_energy.mean()
            epoch_train_sample_energy += sample_energy.mean()
            epoch_train_count         += 1.

        epoch_train_input_energy  /= epoch_train_count
        epoch_train_sample_energy /= epoch_train_count
        train_input_energy.append(epoch_train_input_energy)
        train_sample_energy.append(epoch_train_sample_energy)

        # validation phase
        epoch_valid_input_energy     = 0.
        epoch_valid_sample_energy    = 0.
        epoch_valid_count            = 0.
        valid_batch_iters = valid_stream.get_epoch_iterator()
        for b, valid_batch_data in enumerate(valid_batch_iters):
            # set function inputs
            input_data  = transform(valid_batch_data[0])
            hidden_data = floatX(np_rng.uniform(low=-constant, high=constant, size=(input_data.shape[0], num_hiddens)))
            # evaluate model
            outputs = eval_sample_function(input_data, hidden_data)
            epoch_valid_input_energy  += outputs[0].mean()
            epoch_valid_sample_energy += outputs[1].mean()
            epoch_valid_count         += 1.

        epoch_valid_input_energy  /= epoch_valid_count
        epoch_valid_sample_energy /= epoch_valid_count
        valid_input_energy.append(epoch_valid_input_energy)
        valid_sample_energy.append(epoch_valid_sample_energy)

        print '================================================================'
        print 'EPOCH #{}'.format(e), model_test_name
        print '================================================================'
        print '   TRAIN RESULTS'
        print '================================================================'
        print '     input energy     : ', epoch_train_input_energy
        print '----------------------------------------------------------------'
        print '     sample energy    : ', epoch_train_sample_energy
        print '================================================================'
        print '   VALID RESULTS'
        print '================================================================'
        print '     input energy     : ', epoch_valid_input_energy
        print '----------------------------------------------------------------'
        print '     sample energy    : ', epoch_valid_sample_energy
        print '================================================================'
        print '   OTHERS'
        print '================================================================'
        print '     learning rate    : ', learning_rate
        print '----------------------------------------------------------------'
        print '     noise scale      : ',init_noise*(noise_decay**e)
        print '================================================================'

        # # plot learning curve
        # save_as = model_test_name + '_ENERGY_CURVE.png'
        # plot_learning_curve(cost_values=[train_input_energy,
        #                                  train_sample_energy,
        #                                  valid_input_energy,
        #                                  valid_sample_energy],
        #                     cost_names=['Input Energy (train)',
        #                                 'Sample Energy (train)',
        #                                 'Input Energy (valid)',
        #                                 'Sample Energy (valid)'],
        #                     save_as=save_as)

        save_as = model_test_name + '_SAMPLES{}.png'.format(e+1)
        samples = sampler_function(fixed_hidden_data)[0]
        color_grid_vis(inverse_transform(samples).transpose([0,2,3,1]), (np.sqrt(num_display), np.sqrt(num_display)), save_as)


if __name__=="__main__":
    lr_list          = [1e-4]
    lambda_eng_list  = [1e-5]
    lambda_gen_list  = [1e-5]
    init_noise_list  = [0.01]
    noise_decay_list = [0.98]

    for lr in lr_list:
        for lambda_eng in lambda_eng_list:
            for lambda_gen in lambda_gen_list:
                for init_noise in init_noise_list:
                    for noise_decay in noise_decay_list:
                        train_model(learning_rate= lr,
                                    lambda_eng=lambda_eng,
                                    lambda_gen=lambda_gen,
                                    init_noise=init_noise,
                                    noise_decay=noise_decay,
                                    batch_size=batch_size,
                                    num_epochs=200)