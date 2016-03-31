import os
import numpy as np
from time import time
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
from collections import OrderedDict
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

model_name  = 'ENERGY_RBM_CIFAR10_NEW'
samples_dir = 'samples/%s'%model_name
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

###############
# DATA PARAMS #
###############
num_channels = 3
input_shape  = 32
input_size   = input_shape*input_shape*num_channels
#####################
# INITIALIZE PARAMS #
#####################
filter_size  = 5
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
gain_ifn = Normal(loc=1., scale=0.001)
bias_ifn = Constant(c=0.)
###################
# BUILD GENERATOR #
###################
def set_generator_model(num_hiddens, min_num_gen_filters):
    init_image_size  = 4
    num_gen_filters0 = min_num_gen_filters*4
    num_gen_filters1 = min_num_gen_filters*2
    num_gen_filters2 = min_num_gen_filters*1
    # LAYER 0 (LINEAR)
    linear_w0 = gifn((num_hiddens, num_gen_filters0*init_image_size*init_image_size), 'gen_linear_w0')
    bn_w0     = gain_ifn((num_gen_filters0*init_image_size*init_image_size), 'gen_bn_w0')
    bn_b0     = bias_ifn((num_gen_filters0*init_image_size*init_image_size), 'gen_bn_b0')
    # LAYER 1 (DECONV)
    conv_w1   = gifn((num_gen_filters0, num_gen_filters1, filter_size, filter_size), 'gen_conv_w1')
    bn_w1     = gain_ifn(num_gen_filters1, 'gen_bn_w1')
    bn_b1     = bias_ifn(num_gen_filters1, 'gen_bn_b1')
    # LAYER 2 (DECONV)
    conv_w2   = gifn((num_gen_filters1, num_gen_filters2, filter_size, filter_size), 'gen_conv_w2')
    bn_w2     = gain_ifn(num_gen_filters2, 'gen_bn_w2')
    bn_b2     = bias_ifn(num_gen_filters2, 'gen_bn_b2')
    # LAYER 3 (DECONV)
    conv_w3   = gifn((num_gen_filters2, num_channels, filter_size, filter_size), 'gen_conv_w3')
    conv_b3   = bias_ifn(num_channels, 'gen_conv_b3')
    generator_params = [linear_w0, bn_w0, bn_b0, conv_w1, bn_w1, bn_b1, conv_w2, bn_w2, bn_b2, conv_w3, conv_b3]

    def generator_function(hidden_data, is_train=True):
        h0     = relu(batchnorm(X=T.dot(hidden_data, linear_w0), g=bn_w0, b=bn_b0))
        h0     = h0.reshape((h0.shape[0], num_gen_filters0, init_image_size, init_image_size))
        h1     = relu(batchnorm(deconv(h0, conv_w1, subsample=(2, 2), border_mode=(2, 2)), g=bn_w1, b=bn_b1))
        h2     = relu(batchnorm(deconv(h1, conv_w2, subsample=(2, 2), border_mode=(2, 2)), g=bn_w2, b=bn_b2))
        output = tanh(deconv(h2, conv_w3, subsample=(2, 2), border_mode=(2, 2))+conv_b3.dimshuffle('x', 0, 'x', 'x'))
        return output

    return [generator_function, generator_params]
######################################
# BUILD ENERGY MODEL (FEATURE_MODEL) #
######################################
def set_energy_model(num_hiddens, min_num_eng_filters):
    min_image_size   = 4
    num_eng_filters0 = min_num_eng_filters*1
    num_eng_filters1 = min_num_eng_filters*2
    num_eng_filters2 = min_num_eng_filters*4
    # FEATURE LAYER 0 (DECONV)
    conv_w0   = difn((num_eng_filters0, num_channels, filter_size, filter_size), 'feat_conv_w0')
    conv_b0   = bias_ifn(num_eng_filters0, 'feat_conv_b0')
    # FEATURE LAYER 1 (DECONV)
    conv_w1   = difn((num_eng_filters1, num_eng_filters0, filter_size, filter_size), 'feat_conv_w1')
    conv_b1   = bias_ifn(num_eng_filters1, 'feat_conv_b1')
    # FEATURE LAYER 2 (DECONV)
    conv_w2   = difn((num_eng_filters2, num_eng_filters1, filter_size, filter_size), 'feat_conv_w2')
    conv_b2   = bias_ifn(num_eng_filters2, 'feat_conv_b2')

    def feature_function(input_data, is_train=True):
        # h0 = dropout(relu(dnn_conv(input_data, conv_w0, subsample=(2, 2), border_mode=(2, 2)) + conv_b0.dimshuffle('x', 0, 'x', 'x')), p=0.5, is_training=is_train)
        # h1 = dropout(relu(dnn_conv(        h0, conv_w1, subsample=(2, 2), border_mode=(2, 2)) + conv_b1.dimshuffle('x', 0, 'x', 'x')), p=0.5, is_training=is_train)
        # h2 = dropout(tanh(dnn_conv(        h1, conv_w2, subsample=(2, 2), border_mode=(2, 2)) + conv_b2.dimshuffle('x', 0, 'x', 'x')), p=0.5, is_training=is_train)
        h0 = relu(dnn_conv(input_data, conv_w0, subsample=(2, 2), border_mode=(2, 2)) + conv_b0.dimshuffle('x', 0, 'x', 'x'))
        h1 = relu(dnn_conv(        h0, conv_w1, subsample=(2, 2), border_mode=(2, 2)) + conv_b1.dimshuffle('x', 0, 'x', 'x'))
        h2 = tanh(dnn_conv(        h1, conv_w2, subsample=(2, 2), border_mode=(2, 2)) + conv_b2.dimshuffle('x', 0, 'x', 'x'))
        f  = T.flatten(h2, 2)
        return f


    # ENERGY LAYER (LINEAR)
    feature_mean = bias_ifn(num_eng_filters2*(min_image_size*min_image_size), 'feature_mean')
    feature_std  = bias_ifn(num_eng_filters2*(min_image_size*min_image_size), 'feature_std')
    linear_w0    = difn((num_eng_filters2*(min_image_size*min_image_size), num_hiddens), 'eng_linear_w0')
    linear_b0    = bias_ifn(num_hiddens, 'eng_linear_b0')

    energy_params = [conv_w0, conv_b0, conv_w1, conv_b1, conv_w2, conv_b2, feature_mean, feature_std, linear_w0, linear_b0]

    def energy_function(input_data, is_train=True):
        feature_std_inv = T.inv(T.exp(feature_std)+1e-10)
        e = softplus(T.dot(input_data*feature_std_inv, linear_w0)+linear_b0)
        e = T.sum(-e, axis=1)
        e += 0.5*T.sqr(feature_std_inv)*T.sum(T.sqr(input_data-feature_mean), axis=1, keepdims=True)
        return e

    return [feature_function, energy_function, energy_params]
########################
# ENERGY MODEL UPDATER #
########################
def set_energy_update_function(feature_function,
                               energy_function,
                               generator_function,
                               energy_params,
                               energy_optimizer):

    # set input data, hidden data, annealing rate
    input_data  = T.tensor4(name='input_data',
                            dtype=theano.config.floatX)
    hidden_data = T.matrix(name='hidden_data',
                           dtype=theano.config.floatX)

    annealing = T.scalar(name='annealing',
                         dtype=theano.config.floatX)

    # annealing scale
    annealing_scale = 1.0#/(1.0+99.0*(0.9**annealing))

    # get sample data
    sample_data = generator_function(hidden_data, is_train=True)

    # get feature data
    input_feature  = feature_function(input_data, is_train=True)
    sample_feature = feature_function(sample_data, is_train=True)

    # get energy value
    input_energy  = energy_function(input_feature, is_train=True)
    sample_energy = energy_function(sample_feature, is_train=True)

    # get energy function cost (positive, negative)
    positive_phase      = T.mean(input_energy*annealing_scale)
    negative_phase      = -T.mean(sample_energy*annealing_scale)
    energy_updates_cost = positive_phase + negative_phase

    # get energy updates
    energy_updates = energy_optimizer(energy_params, energy_updates_cost)

    # update function input
    update_function_inputs  = [input_data,
                               hidden_data,
                               annealing]

    # update function output
    update_function_outputs = [input_energy,
                               sample_energy]

    # update function
    update_function = theano.function(inputs=update_function_inputs,
                                      outputs=update_function_outputs,
                                      updates=energy_updates,
                                      on_unused_input='ignore')
    return update_function
#####################
# GENERATOR UPDATER #
#####################
def set_generator_update_function(feature_function,
                                  energy_function,
                                  generator_function,
                                  generator_params,
                                  generator_optimizer):

    # set input data, hidden data, noise_data annealing rate
    hidden_data = T.matrix(name='hidden_data',
                           dtype=theano.config.floatX)
    noise_data  = T.tensor4(name='noise_data',
                            dtype=theano.config.floatX)
    annealing = T.scalar(name='annealing',
                         dtype=theano.config.floatX)

    # annealing scale
    annealing_scale = 1.0#/(1.0+99.0*(0.9**annealing))

    # get sample data
    sample_data = generator_function(hidden_data, is_train=True)
    sample_data = T.clip(sample_data+noise_data, -1., 1.)

    # get feature data
    sample_feature = feature_function(sample_data, is_train=True)

    # get energy value
    sample_energy = energy_function(sample_feature, is_train=True)

    # get generator update cost
    generator_updates_cost = T.mean(sample_energy*annealing_scale)

    # get generator updates
    generator_updates = generator_optimizer(generator_params, generator_updates_cost)

    # update function input
    update_function_inputs  = [hidden_data,
                               annealing]

    # update function output
    update_function_outputs = [sample_energy,]

    # update function
    update_function = theano.function(inputs=update_function_inputs,
                                      outputs=update_function_outputs,
                                      updates=generator_updates,
                                      on_unused_input='ignore')
    return update_function
#############
# EVALUATOR #
#############
def set_evaluation_and_sampling_function(feature_function,
                                         energy_function,
                                         generator_function):
    # set input data, hidden data, annealing rate
    input_data  = T.tensor4(name='input_data',
                            dtype=theano.config.floatX)
    hidden_data = T.matrix(name='hidden_data',
                           dtype=theano.config.floatX)

    # get sample data
    sample_data = generator_function(hidden_data, is_train=False)

    # get feature data
    input_feature  = feature_function(input_data, is_train=False)
    sample_feature = feature_function(sample_data, is_train=False)

    # get energy value
    input_energy  = energy_function(input_feature, is_train=False)
    sample_energy = energy_function(sample_feature, is_train=False)

    function_inputs = [input_data,
                       hidden_data]
    function_outputs = [input_energy,
                        sample_energy]

    function = theano.function(inputs=function_inputs,
                               outputs=function_outputs,
                               on_unused_input='ignore')
    return function
###########
# SAMPLER #
###########
def set_sampling_train_function(generator_function):

    hidden_data = T.matrix(name='hidden_data',
                           dtype=theano.config.floatX)

    sample_data = generator_function(hidden_data, is_train=True)

    function_inputs = [hidden_data,]
    function_outputs = [sample_data,]

    function = theano.function(inputs=function_inputs,
                               outputs=function_outputs,
                               on_unused_input='ignore')
    return function
###########
# SAMPLER #
###########
def set_sampling_test_function(generator_function):

    hidden_data = T.matrix(name='hidden_data',
                           dtype=theano.config.floatX)

    sample_data = generator_function(hidden_data, is_train=False)

    function_inputs = [hidden_data,]
    function_outputs = [sample_data,]

    function = theano.function(inputs=function_inputs,
                               outputs=function_outputs,
                               on_unused_input='ignore')
    return function
###########
# TRAINER #
###########
def train_model(train_stream,
                valid_stream,
                energy_optimizer,
                generator_optimizer,
                model_config_dict,
                model_test_name):

    [generator_function, generator_params] = set_generator_model(model_config_dict['hidden_size'],
                                                                 model_config_dict['min_num_gen_filters'])
    [feature_function, energy_function, energy_params] = set_energy_model(model_config_dict['hidden_size'],
                                                                          model_config_dict['min_num_eng_filters'])
    # compile functions
    print 'COMPILING ENERGY UPDATER'
    t=time()
    energy_updater = set_energy_update_function(feature_function=feature_function,
                                                energy_function=energy_function,
                                                generator_function=generator_function,
                                                energy_params=energy_params,
                                                energy_optimizer=energy_optimizer)
    print '%.2f SEC '%(time()-t)
    print 'COMPILING GENERATOR UPDATER'
    t=time()
    generator_updater = set_generator_update_function(feature_function=feature_function,
                                                      energy_function=energy_function,
                                                      generator_function=generator_function,
                                                      generator_params=generator_params,
                                                      generator_optimizer=generator_optimizer)
    print '%.2f SEC '%(time()-t)
    print 'COMPILING EVALUATION FUNCTION'
    t=time()
    evaluation_function = set_evaluation_and_sampling_function(feature_function=feature_function,
                                                               energy_function=energy_function,
                                                               generator_function=generator_function)
    print '%.2f SEC '%(time()-t)
    print 'COMPILING SAMPLING FUNCTION'
    t=time()
    sampling_train_function = set_sampling_train_function(generator_function=generator_function)
    sampling_test_function  = set_sampling_test_function(generator_function=generator_function)
    print '%.2f SEC '%(time()-t)

    # set fixed hidden data for sampling
    fixed_hidden_data  = floatX(np_rng.uniform(low=-model_config_dict['hidden_distribution'],
                                               high=model_config_dict['hidden_distribution'],
                                               size=(model_config_dict['num_display'], model_config_dict['hidden_size'])))

    print 'START TRAINING'

    # train output
    train_input_energy  = []
    train_sample_energy = []
    # valid output
    valid_input_energy  = []
    valid_sample_energy = []

    # for each epoch
    for e in xrange(model_config_dict['epochs']):
        # train phase
        epoch_train_input_energy  = 0.
        epoch_train_sample_energy = 0.
        epoch_train_count         = 0.

        train_batch_iters = train_stream.get_epoch_iterator()
        # for each batch
        for b, train_batch_data in enumerate(train_batch_iters):
            # set update function inputs
            input_data   = transform(train_batch_data[0])
            hidden_data  = floatX(np_rng.uniform(low=-model_config_dict['hidden_distribution'],
                                                 high=model_config_dict['hidden_distribution'],
                                                 size=(model_config_dict['batch_size'], model_config_dict['hidden_size'])))
            noise_data   = floatX(np_rng.normal(size=input_data.shape))
            noise_data   = noise_data*model_config_dict['init_noise']*(model_config_dict['noise_decay']**e)

            # update generator
            generator_update_inputs = [hidden_data,
                                       noise_data,
                                       e]
            [sample_energy, ] = generator_updater(*generator_update_inputs)

            # update energy function
            energy_update_inputs = [input_data,
                                    hidden_data,
                                    e]
            [input_energy, sample_energy, ] = energy_updater(*energy_update_inputs)

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
            hidden_data  = floatX(np_rng.uniform(low=-model_config_dict['hidden_distribution'],
                                                 high=model_config_dict['hidden_distribution'],
                                                 size=(model_config_dict['batch_size'], model_config_dict['hidden_size'])))
            # evaluate model
            evaluation_input = [input_data,  hidden_data]
            outputs = evaluation_function(*evaluation_input)
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

        # plot curve data
        save_as = model_test_name + '_ENERGY_CURVE.png'
        plot_learning_curve(cost_values=[train_input_energy,
                                         train_sample_energy,
                                         valid_input_energy,
                                         valid_sample_energy],
                            cost_names=['Input Energy (train)',
                                        'Sample Energy (train)',
                                        'Input Energy (valid)',
                                        'Sample Energy (valid)'],
                            save_as=save_as)

        # sample data
        save_as = samples_dir + '/' + model_test_name + '_TRAIN_SAMPLES{}.png'.format(e+1)
        train_sample_data = sampling_train_function(fixed_hidden_data)[0]
        train_sample_data = np.asarray(train_sample_data)
        color_grid_vis(inverse_transform(train_sample_data).transpose([0,2,3,1]), (16, 16), save_as)

        save_as = samples_dir + '/' + model_test_name + '_TEST_SAMPLES{}.png'.format(e+1)
        test_sample_data  = sampling_test_function(fixed_hidden_data)[0]
        test_sample_data  = np.asarray(test_sample_data)
        color_grid_vis(inverse_transform(test_sample_data).transpose([0,2,3,1]), (16, 16), save_as)

if __name__=="__main__":

    model_config_dict = OrderedDict()
    model_config_dict['batch_size']          = 128
    model_config_dict['num_display']         = 16*16
    model_config_dict['hidden_distribution'] = 1.
    model_config_dict['epochs']              = 100

    #################
    # LOAD DATA SET #
    #################
    train_data, test_data, train_stream, valid_stream, test_stream = cifar10(window_size=(input_shape, input_shape), batch_size=model_config_dict['batch_size'])

    hidden_size_list = [100, 1000]
    num_filters_list = [32, 16]
    lr_list          = [1e-3]
    dropout_list     = [False, ]
    lambda_eng_list  = [1e-5, 1e-10]
    lambda_gen_list  = [1e-5, 1e-10]
    init_noise_list  = [1e-2, 1e-3]
    noise_decay_list = [1.00, 0.98]

    for lr in lr_list:
        for num_filters in num_filters_list:
            for hidden_size in hidden_size_list:
                for dropout in dropout_list:
                    for lambda_eng in lambda_eng_list:
                        for lambda_gen in lambda_gen_list:
                            for init_noise in init_noise_list:
                                for noise_decay in noise_decay_list:
                                    model_config_dict['hidden_size']         = hidden_size
                                    model_config_dict['min_num_gen_filters'] = num_filters
                                    model_config_dict['min_num_eng_filters'] = num_filters
                                    model_config_dict['init_noise']          = init_noise
                                    model_config_dict['noise_decay']         = noise_decay

                                    # set updates
                                    energy_optimizer    = Adagrad(lr=sharedX(lr),
                                                                  regularizer=Regularizer(l2=lambda_eng))
                                    generator_optimizer = Adagrad(lr=sharedX(lr),
                                                                  regularizer=Regularizer(l2=lambda_gen))
                                    model_test_name = model_name \
                                                      + '_f{}'.format(int(num_filters)) \
                                                      + '_h{}'.format(int(hidden_size)) \
                                                      + '_d{}'.format(int(dropout)) \
                                                      + '_re{}'.format(int(-np.log10(lambda_eng))) \
                                                      + '_rg{}'.format(int(-np.log10(lambda_gen))) \
                                                      + '_n{}'.format(int(-np.log10(init_noise))) \
                                                      + '_d{}'.format(int(1 if noise_decay is 1.0 else 0)) \

                                    train_model(train_stream=train_stream,
                                                valid_stream=valid_stream,
                                                energy_optimizer=energy_optimizer,
                                                generator_optimizer=generator_optimizer,
                                                model_config_dict=model_config_dict,
                                                model_test_name=model_test_name)
