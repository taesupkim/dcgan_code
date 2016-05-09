import os
import numpy as np
from time import time
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
from collections import OrderedDict
from lib.activations import Rectify, Tanh, Softplus, LeakyRectify
from lib.updates import Adagrad, Regularizer, RMSprop, Adam
from lib.inits import Normal, Constant
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, entropykeep, deconv, dropout, l2normalize
from lib.theano_utils import floatX, sharedX
from load import imagenet
from lib.save_utils import save_model, unpickle

def transform(X):
    return floatX(X)/127.5 - 1.

def inverse_transform(X):
    X = (X+1.)/2.
    return X

###################
# SET INITIALIZER #
###################
def get_entropy_cost(entropy_params_list):
    entropy_const = 0.5*(1.0+np.log(2.0*np.pi))
    entropy_const = entropy_const.astype(theano.config.floatX)

    entropy_tensor_params= []
    for entropy_params in entropy_params_list:
        entropy_tensor_params.append(entropy_params.reshape((1,-1)))
    entropy_tensor_params = T.concatenate(entropy_tensor_params, axis=1)
    entropy_tensor_params = 0.5*T.log(T.sqr(entropy_tensor_params))
    entropy_cost = T.sum(-entropy_const-entropy_tensor_params)
    return entropy_cost

model_name  = 'ENERGY_RBM_IMAGENET_BIAS_ADAGRAD_NORMED'
samples_dir = 'samples/%s'%model_name
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

###############
# DATA PARAMS #
###############
num_channels = 3
input_shape  = 64
input_size   = input_shape*input_shape*num_channels

#####################
# INITIALIZE PARAMS #
#####################
filter_size  = 5
filter_shape = (filter_size, filter_size)

##############################
# SET ACTIVATIONS AND OTHERS #
##############################
leak_relu = LeakyRectify()
relu  = Rectify()
tanh  = Tanh()
softplus = Softplus()

###################
# SET INITIALIZER #
###################
weight_init = Normal(scale=0.01)
scale_ones  = Constant(c=1.0)
scale_init  = Constant(c=1.0)
bias_zero   = Constant(c=0.0)
bias_const  = Constant(c=0.1)

###################
# BUILD GENERATOR #
###################
def set_generator_model(num_hiddens,
                        min_num_gen_filters):
    # initial square image size
    init_image_size  = 4
    
    # set num of filters for each layer
    num_gen_filters0 = min_num_gen_filters*8
    num_gen_filters1 = min_num_gen_filters*4
    num_gen_filters2 = min_num_gen_filters*2
    num_gen_filters3 = min_num_gen_filters*1

    # LAYER 0 (LINEAR W/ BN)
    print 'SET GENERATOR LINEAR LAYER 0'
    linear_w0    = weight_init((num_hiddens,
                                (num_gen_filters0*init_image_size*init_image_size)),
                               'gen_linear_w0')
    linear_bn_w0 = scale_ones((num_gen_filters0*init_image_size*init_image_size),
                              'gen_linear_bn_w0')
    linear_bn_b0 = bias_const((num_gen_filters0*init_image_size*init_image_size),
                              'gen_linear_bn_b0')

    # LAYER 1 (DECONV)
    print 'SET GENERATOR CONV LAYER 1'
    conv_w1    = weight_init((num_gen_filters0, num_gen_filters1) + filter_shape,
                             'gen_conv_w1')
    conv_bn_w1 = scale_ones(num_gen_filters1,
                            'gen_conv_bn_w1')
    conv_bn_b1 = bias_const(num_gen_filters1,
                            'gen_conv_bn_b1')

    # LAYER 2 (DECONV)
    print 'SET GENERATOR CONV LAYER 2'
    conv_w2    = weight_init((num_gen_filters1, num_gen_filters2) + filter_shape,
                             'gen_conv_w2')
    conv_bn_w2 = scale_ones(num_gen_filters2,
                            'gen_conv_bn_w2')
    conv_bn_b2 = bias_const(num_gen_filters2,
                            'gen_conv_bn_b2')

    # LAYER 2 (DECONV)
    print 'SET GENERATOR CONV LAYER 3'
    conv_w3    = weight_init((num_gen_filters2, num_gen_filters3) + filter_shape,
                             'gen_conv_w3')
    conv_bn_w3 = scale_ones(num_gen_filters3,
                            'gen_conv_bn_w3')
    conv_bn_b3 = bias_const(num_gen_filters3,
                            'gen_conv_bn_b3')

    # LAYER 3 (DECONV)
    print 'SET GENERATOR CONV LAYER 4'
    conv_w4 = weight_init((num_gen_filters3, num_channels) + filter_shape,
                          'gen_conv_w4')
    conv_b4 = bias_zero(num_channels,
                        'gen_conv_b4')

    generator_params = [[linear_w0, linear_bn_b0,
                         conv_w1, conv_bn_b1,
                         conv_w2, conv_bn_b2,
                         conv_w3, conv_bn_b3,
                         conv_w4, conv_b4],
                        [linear_bn_w0,
                         conv_bn_w1,
                         conv_bn_w2,
                         conv_bn_w3]]

    print 'SET GENERATOR FUNCTION'
    def generator_function(hidden_data, is_train=True):
        # layer 0 (linear)
        h0     = relu(batchnorm(X=T.dot(hidden_data, linear_w0), g=linear_bn_w0, b=linear_bn_b0))
        h0     = h0.reshape((h0.shape[0], num_gen_filters0, init_image_size, init_image_size))
        # layer 1 (deconv)
        h1     = relu(batchnorm(deconv(h0, conv_w1, subsample=(2, 2), border_mode=(2, 2)), g=conv_bn_w1, b=conv_bn_b1))
        # layer 2 (deconv)
        h2     = relu(batchnorm(deconv(h1, conv_w2, subsample=(2, 2), border_mode=(2, 2)), g=conv_bn_w2, b=conv_bn_b2))
        # layer 3 (deconv)
        h3     = batchnorm(deconv(h2, conv_w3, subsample=(2, 2), border_mode=(2, 2)), g=conv_bn_w3, b=conv_bn_b3)
        # layer 4 (deconv)
        output = tanh(deconv(h3, conv_w4, subsample=(2, 2), border_mode=(2, 2))+conv_b4.dimshuffle('x', 0, 'x', 'x'))
        return output

    return [generator_function, generator_params]

##################
# LOAD GENERATOR #
##################
def load_generator_model(min_num_gen_filters,
                         model_params_dict):
    # initial square image size
    init_image_size  = 4

    # set num of filters for each layer
    num_gen_filters0 = min_num_gen_filters*8

    # LAYER 0 (LINEAR W/ BN)
    print 'LOAD GENERATOR LINEAR LAYER 0'
    linear_w0    = sharedX(model_params_dict[   'gen_linear_w0'], name='gen_linear_w0')
    linear_bn_w0 = sharedX(model_params_dict['gen_linear_bn_w0'], name='gen_linear_bn_w0')
    linear_bn_b0 = sharedX(model_params_dict['gen_linear_bn_b0'], name='gen_linear_bn_b0')

    # LAYER 1 (DECONV)
    print 'SET GENERATOR CONV LAYER 1'
    conv_w1    = sharedX(model_params_dict[   'gen_conv_w1'], name='gen_conv_w1')
    conv_bn_w1 = sharedX(model_params_dict['gen_conv_bn_w1'], name='gen_conv_bn_w1')
    conv_bn_b1 = sharedX(model_params_dict['gen_conv_bn_b1'], name='gen_conv_bn_b1')

    # LAYER 2 (DECONV)
    print 'SET GENERATOR CONV LAYER 2'
    conv_w2    = sharedX(model_params_dict[   'gen_conv_w2'], name='gen_conv_w2')
    conv_bn_w2 = sharedX(model_params_dict['gen_conv_bn_w2'], name='gen_conv_bn_w2')
    conv_bn_b2 = sharedX(model_params_dict['gen_conv_bn_b2'], name='gen_conv_bn_b2')

    # LAYER 2 (DECONV)
    print 'SET GENERATOR CONV LAYER 3'
    conv_w3    = sharedX(model_params_dict[   'gen_conv_w3'], name='gen_conv_w3')
    conv_bn_w3 = sharedX(model_params_dict['gen_conv_bn_w3'], name='gen_conv_bn_w3')
    conv_bn_b3 = sharedX(model_params_dict['gen_conv_bn_b3'], name='gen_conv_bn_b3')

    # LAYER 3 (DECONV)
    print 'SET GENERATOR CONV LAYER 4'
    conv_w4 = sharedX(model_params_dict[   'gen_conv_w4'], name='gen_conv_w4')
    conv_b4 = sharedX(model_params_dict[   'gen_conv_b4'], name='gen_conv_b4')

    generator_params = [[linear_w0, linear_bn_b0,
                         conv_w1, conv_bn_b1,
                         conv_w2, conv_bn_b2,
                         conv_w3, conv_bn_b3,
                         conv_w4, conv_b4],
                        [linear_bn_w0,
                         conv_bn_w1,
                         conv_bn_w2,
                         conv_bn_w3]]

    print 'SET GENERATOR FUNCTION'
    def generator_function(hidden_data, is_train=True):
        # layer 0 (linear)
        h0     = relu(batchnorm(X=T.dot(hidden_data, linear_w0), g=linear_bn_w0, b=linear_bn_b0))
        h0     = h0.reshape((h0.shape[0], num_gen_filters0, init_image_size, init_image_size))
        # layer 1 (deconv)
        h1     = relu(batchnorm(deconv(h0, conv_w1, subsample=(2, 2), border_mode=(2, 2)), g=conv_bn_w1, b=conv_bn_b1))
        # layer 2 (deconv)
        h2     = relu(batchnorm(deconv(h1, conv_w2, subsample=(2, 2), border_mode=(2, 2)), g=conv_bn_w2, b=conv_bn_b2))
        # layer 3 (deconv)
        h3     = batchnorm(deconv(h2, conv_w3, subsample=(2, 2), border_mode=(2, 2)), g=conv_bn_w3, b=conv_bn_b3)
        # layer 4 (deconv)
        output = tanh(deconv(h3, conv_w4, subsample=(2, 2), border_mode=(2, 2))+conv_b4.dimshuffle('x', 0, 'x', 'x'))
        return output

    return [generator_function, generator_params]
######################################
# BUILD ENERGY MODEL (FEATURE_MODEL) #
######################################
def set_energy_model(num_experts,
                     min_num_eng_filters):

    # minimum square image size
    min_image_size   = 4

    # set num of filters for each layer
    num_eng_filters0 = min_num_eng_filters*1
    num_eng_filters1 = min_num_eng_filters*2
    num_eng_filters2 = min_num_eng_filters*4
    num_eng_filters3 = min_num_eng_filters*8

    # FEATURE LAYER 0 (DECONV)
    print 'SET ENERGY FEATURE CONV LAYER 0'
    conv_w0   = weight_init((num_eng_filters0, num_channels) + filter_shape,
                            'feat_conv_w0')
    conv_b0   = bias_const(num_eng_filters0,
                           'feat_conv_b0')
    # FEATURE LAYER 1 (DECONV)
    print 'SET ENERGY FEATURE CONV LAYER 1'
    conv_w1   = weight_init((num_eng_filters1, num_eng_filters0) + filter_shape,
                            'feat_conv_w1')
    conv_b1   = bias_const(num_eng_filters1,
                           'feat_conv_b1')
    # FEATURE LAYER 2 (DECONV)
    print 'SET ENERGY FEATURE CONV LAYER 2'
    conv_w2   = weight_init((num_eng_filters2, num_eng_filters1) + filter_shape,
                            'feat_conv_w2')
    conv_b2   = bias_const(num_eng_filters2,
                           'feat_conv_b2')
    # FEATURE LAYER 3 (DECONV)
    print 'SET ENERGY FEATURE CONV LAYER 3'
    conv_w3   = weight_init((num_eng_filters3, num_eng_filters2) + filter_shape,
                            'feat_conv_w3')

    print 'SET ENERGY FEATURE EXTRACTOR'
    def feature_function(input_data, is_train=True):
        # layer 0 (conv)
        h0 = relu(dnn_conv(input_data, conv_w0, subsample=(2, 2), border_mode=(2, 2))+conv_b0.dimshuffle('x', 0, 'x', 'x'))
        # layer 1 (conv)
        h1 = relu(dnn_conv(        h0, conv_w1, subsample=(2, 2), border_mode=(2, 2))+conv_b1.dimshuffle('x', 0, 'x', 'x'))
        # layer 2 (conv)
        h2 = relu(dnn_conv(        h1, conv_w2, subsample=(2, 2), border_mode=(2, 2))+conv_b2.dimshuffle('x', 0, 'x', 'x'))
        # layer 3 (conv)
        h3 = dnn_conv(        h2, conv_w3, subsample=(2, 2), border_mode=(2, 2))
        feature = T.flatten(h3, 2)
        return feature

    # ENERGY LAYER (LINEAR)
    print 'SET ENERGY FUNCTION LINEAR LAYER 4'
    linear_w4 = weight_init((num_eng_filters3*(min_image_size*min_image_size),
                             num_experts),
                            'eng_linear_w4')
    linear_b4 = bias_zero(num_experts,
                          'eng_linear_b4')

    energy_params = [conv_w0, conv_b0,
                     conv_w1, conv_b1,
                     conv_w2, conv_b2,
                     conv_w3,
                     linear_w4, linear_b4]

    def energy_function(feature_data, is_train=True):
        e = softplus(T.dot(feature_data, linear_w4)+linear_b4)
        e = T.sum(-e, axis=1, keepdims=True)
        e += 0.5*T.sum(T.sqr(feature_data), axis=1, keepdims=True)
        return e

    return [feature_function, energy_function, energy_params]

def set_model_update_function(energy_feature_function,
                              energy_function,
                              generator_function,
                              energy_params,
                              generator_params,
                              energy_optimizer,
                              generator_optimizer):

    # set input data, hidden data, noise data,  annealing rate
    input_data  = T.tensor4(name='input_data',
                            dtype=theano.config.floatX)
    hidden_data = T.matrix(name='hidden_data',
                           dtype=theano.config.floatX)
    noise_data  = T.tensor4(name='noise_data',
                            dtype=theano.config.floatX)

    # get sample data
    sample_data = generator_function(hidden_data, is_train=True)
    sample_data = T.clip(sample_data+noise_data, -1.+1e-5, 1.-1e-5)

    # get feature data
    input_feature  = energy_feature_function(input_data, is_train=True)
    sample_feature = energy_feature_function(sample_data, is_train=True)

    # normalize feature data
    full_feature   = T.concatenate([input_feature, sample_feature], axis=0)
    full_feature   = batchnorm(full_feature)
    input_feature  = full_feature[:input_feature.shape[0]]
    sample_feature = full_feature[input_feature.shape[0]:]

    # get energy value
    input_energy  = energy_function(input_feature, is_train=True)
    sample_energy = energy_function(sample_feature, is_train=True)

    # get energy function cost (positive, negative)
    positive_phase      = T.mean(input_energy)
    negative_phase      = T.mean(sample_energy)
    energy_updates_cost = positive_phase - negative_phase

    # get energy updates
    energy_updates = energy_optimizer(energy_params,
                                      energy_updates_cost)

    # entropy cost
    entropy_cost = get_entropy_cost(generator_params[1])

    # entropy weight
    entropy_weights = []
    for param_tensor in generator_params[1]:
        entropy_weights.append(param_tensor.reshape((1,-1)))
    entropy_weights = T.concatenate(entropy_weights, axis=1)
    entropy_weights = T.abs_(entropy_weights)
    entropy_weights = T.mean(entropy_weights)

    # get generator updates
    generator_updates_cost = negative_phase + entropy_cost
    generator_updates = generator_optimizer(generator_params[0]+generator_params[1],
                                            generator_updates_cost)

    # update function input
    update_function_inputs  = [input_data,
                               hidden_data,
                               noise_data]

    # update function output
    update_function_outputs = [input_energy,
                               sample_energy,
                               entropy_weights,
                               entropy_cost]

    # update function
    update_function = theano.function(inputs=update_function_inputs,
                                      outputs=update_function_outputs,
                                      updates=energy_updates+generator_updates,
                                      on_unused_input='ignore')
    return update_function


########################
# ENERGY MODEL UPDATER #
########################
def set_energy_update_function(feature_function,
                               energy_function,
                               generator_function,
                               energy_params,
                               energy_optimizer):

    # set input data, hidden data, noise data,  annealing rate
    input_data  = T.tensor4(name='input_data',
                            dtype=theano.config.floatX)
    hidden_data = T.matrix(name='hidden_data',
                           dtype=theano.config.floatX)
    # get sample data
    sample_data = generator_function(hidden_data, is_train=True)

    # get feature data
    input_feature  = feature_function(input_data, is_train=True)
    sample_feature = feature_function(sample_data, is_train=True)

    # get energy value
    input_energy  = energy_function(input_feature, is_train=True)
    sample_energy = energy_function(sample_feature, is_train=True)

    # get energy function cost (positive, negative)
    positive_phase      = T.mean(input_energy)
    negative_phase      = T.mean(sample_energy)
    energy_updates_cost = positive_phase - negative_phase

    # get energy updates
    energy_updates = energy_optimizer(energy_params,
                                      energy_updates_cost)

    # update function input
    update_function_inputs  = [input_data,
                               hidden_data]

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
                                  generator_entropy_params,
                                  generator_optimizer,
                                  generator_entropy_optimizer):

    # set hidden data
    hidden_data = T.matrix(name='hidden_data',
                           dtype=theano.config.floatX)
    # set noise data
    noise_data  = T.tensor4(name='noise_data',
                            dtype=theano.config.floatX)
    # get sample data
    sample_data = generator_function(hidden_data, is_train=True)
    sample_data = T.clip(sample_data+noise_data, -1.+1e-5, 1.-1e-5)

    # get feature data
    sample_feature = feature_function(sample_data, is_train=True)

    # get energy value
    sample_energy = energy_function(sample_feature, is_train=True)

    # entropy cost
    entropy_cost = get_entropy_cost(generator_entropy_params)

    # entropy weight
    entropy_weights = []
    for param_tensor in generator_entropy_params:
        entropy_weights.append(param_tensor.reshape((1,-1)))
    entropy_weights = T.concatenate(entropy_weights, axis=1)
    entropy_weights = T.abs_(entropy_weights)
    entropy_weights = T.mean(entropy_weights)

    # get generator update cost
    negative_phase         = T.mean(sample_energy)
    generator_updates_cost = negative_phase + entropy_cost

    # get generator updates
    generator_updates = generator_optimizer(generator_params,
                                            generator_updates_cost)

    generator_entropy_updates = generator_entropy_optimizer(generator_entropy_params,
                                                            generator_updates_cost)

    # update function input
    update_function_inputs  = [hidden_data,
                               noise_data]

    # update function output
    update_function_outputs = [sample_energy,
                               entropy_cost,
                               entropy_weights]

    # update function
    update_function = theano.function(inputs=update_function_inputs,
                                      outputs=update_function_outputs,
                                      updates=generator_updates+generator_entropy_updates,
                                      on_unused_input='ignore')
    return update_function

#######################
# VALIDATION FUNCTION #
#######################
def set_validation_function(feature_function,
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
def set_sampling_function(generator_function):

    hidden_data = T.matrix(name='hidden_data',
                           dtype=theano.config.floatX)

    sample_data_t = generator_function(hidden_data, is_train=False)

    function_inputs = [hidden_data,]
    function_outputs = [sample_data_t,]

    function = theano.function(inputs=function_inputs,
                               outputs=function_outputs,
                               on_unused_input='ignore')
    return function

###########
# TRAINER #
###########
def train_model(data_stream,
                energy_optimizer,
                generator_optimizer,
                model_config_dict,
                model_test_name):

    generator_models = set_generator_model(num_hiddens=model_config_dict['hidden_size'],
                                           min_num_gen_filters=model_config_dict['min_num_gen_filters'])
    generator_function       = generator_models[0]
    generator_params         = generator_models[1]
    energy_models = set_energy_model(num_experts=model_config_dict['expert_size'],
                                     min_num_eng_filters=model_config_dict['min_num_eng_filters'])
    feature_function = energy_models[0]
    energy_function  = energy_models[1]
    energy_params    = energy_models[2]

    # compile functions
    print 'COMPILING MODEL UPDATER'
    t=time()
    model_updater = set_model_update_function(energy_feature_function=feature_function,
                                              energy_function=energy_function,
                                              generator_function=generator_function,
                                              energy_params=energy_params,
                                              generator_params=generator_params,
                                              energy_optimizer=energy_optimizer,
                                              generator_optimizer=generator_optimizer)
    print '%.2f SEC '%(time()-t)
    # print 'COMPILING ENERGY UPDATER'
    # t=time()
    # energy_updater = set_energy_update_function(feature_function=feature_function,
    #                                             energy_function=energy_function,
    #                                             generator_function=generator_function,
    #                                             energy_params=energy_params,
    #                                             energy_optimizer=energy_optimizer)
    # print '%.2f SEC '%(time()-t)
    # print 'COMPILING GENERATOR UPDATER'
    # t=time()
    # generator_updater = set_generator_update_function(feature_function=feature_function,
    #                                                   energy_function=energy_function,
    #                                                   generator_function=generator_function,
    #                                                   generator_params=generator_params,
    #                                                   generator_entropy_params=generator_entropy_params,
    #                                                   generator_optimizer=generator_optimizer,
    #                                                   generator_entropy_optimizer=generator_entropy_optimizer)
    # print '%.2f SEC '%(time()-t)

    print 'COMPILING SAMPLING FUNCTION'
    t=time()
    sampling_function = set_sampling_function(generator_function=generator_function)
    print '%.2f SEC '%(time()-t)

    # set fixed hidden data for sampling
    fixed_hidden_data  = floatX(np_rng.uniform(low=-model_config_dict['hidden_distribution'],
                                               high=model_config_dict['hidden_distribution'],
                                               size=(model_config_dict['num_display'], model_config_dict['hidden_size'])))

    print 'START TRAINING'
    # for each epoch
    input_energy_list = []
    sample_energy_list = []
    batch_count = 0
    for e in xrange(model_config_dict['epochs']):
        # train phase
        batch_iters = data_stream.get_epoch_iterator()
        # for each batch
        for b, batch_data in enumerate(batch_iters):
            # set update function inputs
            input_data   = transform(batch_data[0])
            num_data     = input_data.shape[0]
            hidden_data  = floatX(np_rng.uniform(low=-model_config_dict['hidden_distribution'],
                                                 high=model_config_dict['hidden_distribution'],
                                                 size=(num_data, model_config_dict['hidden_size'])))
            noise_data   = floatX(np_rng.normal(scale=0.01,
                                                size=input_data.shape))

            # generator update
            # generator_updates = generator_updater(hidden_data, noise_data)
            # entropy_cost      = generator_updates[-2]
            # entropy_weights   = generator_updates[-1]
            # energy_updates    = energy_updater(input_data, hidden_data)

            # get output values
            # input_energy  = energy_updates[0].mean()
            # sample_energy = energy_updates[1].mean()

            update_input  = [input_data, hidden_data, noise_data]
            update_output = model_updater(*update_input)

            input_energy    = update_output[0].mean()
            sample_energy   = update_output[1].mean()
            entropy_weights = update_output[2].mean()
            entropy_cost    = update_output[3].mean()

            input_energy_list.append(input_energy)
            sample_energy_list.append(sample_energy)

            # batch count up
            batch_count += 1

            if batch_count%10==0:
                print '================================================================'
                print 'BATCH ITER #{}'.format(batch_count), model_test_name
                print '================================================================'
                print '   TRAIN RESULTS'
                print '================================================================'
                print '     input energy     : ', input_energy_list[-1]
                print '----------------------------------------------------------------'
                print '     sample energy    : ', sample_energy_list[-1]
                print '----------------------------------------------------------------'
                print '     entropy weight   : ', entropy_weights
                print '----------------------------------------------------------------'
                print '     entropy cost     : ', entropy_cost
                print '================================================================'

            if batch_count%100==0:
                # sample data
                sample_data = sampling_function(fixed_hidden_data)[0]
                sample_data = np.asarray(sample_data)
                save_as = samples_dir + '/' + model_test_name + '_SAMPLES(TRAIN){}.png'.format(batch_count)
                color_grid_vis(inverse_transform(sample_data).transpose([0,2,3,1]), (16, 16), save_as)
                np.save(file=samples_dir + '/' + model_test_name +'_input_energy',
                        arr=np.asarray(input_energy_list))
                np.save(file=samples_dir + '/' + model_test_name +'_sample_energy',
                        arr=np.asarray(sample_energy_list))

                save_as = samples_dir + '/' + model_test_name + '_MODEL.pkl'
                save_model(tensor_params_list=generator_params[0] + generator_params[1] + energy_params,
                           save_to=save_as)
##########
# TESTER #
##########
def test_model(model_config_dict, model_test_name):
    import glob
    model_list = glob.glob(samples_dir +'/*.pkl')
    # load parameters
    model_param_dicts = unpickle(model_list[0])

    # load generator
    generator_models = load_generator_model(min_num_gen_filters=model_config_dict['min_num_gen_filters'],
                                            model_params_dict=model_param_dicts)
    generator_function = generator_models[0]

    print 'COMPILING SAMPLING FUNCTION'
    t=time()
    sampling_function = set_sampling_function(generator_function=generator_function)
    print '%.2f SEC '%(time()-t)

    print 'START SAMPLING'
    for s in xrange(model_config_dict['num_sampling']):
        print '{} sampling'.format(s)
        hidden_data  = floatX(np_rng.uniform(low=-model_config_dict['hidden_distribution'],
                                             high=model_config_dict['hidden_distribution'],
                                             size=(model_config_dict['num_display'], model_config_dict['hidden_size'])))
        sample_data = sampling_function(hidden_data)[0]
        sample_data = inverse_transform(np.asarray(sample_data)).transpose([0,2,3,1])
        save_as = samples_dir + '/' + model_test_name + '_SAMPLES(TRAIN){}.png'.format(s+1)
        color_grid_vis(sample_data, (16, 16), save_as)

if __name__=="__main__":

    model_config_dict = OrderedDict()
    model_config_dict['batch_size']          = 128
    model_config_dict['num_display']         = 16*16
    model_config_dict['hidden_distribution'] = 1.
    model_config_dict['epochs']              = 200

    is_training = True
    if is_training is True:
        #################
        # LOAD DATA SET #
        #################
        _ , data_stream = imagenet(batch_size=model_config_dict['batch_size'])

        expert_size_list = [8192]
        hidden_size_list = [1024]
        num_filters_list = [256]
        lr_list          = [1e-4]
        lambda_eng_list  = [1e-5]

        for lr in lr_list:
            for num_filters in num_filters_list:
                for hidden_size in hidden_size_list:
                    for expert_size in expert_size_list:
                        for lambda_eng in lambda_eng_list:
                            model_config_dict['hidden_size']         = hidden_size
                            model_config_dict['expert_size']         = expert_size
                            model_config_dict['min_num_gen_filters'] = num_filters
                            model_config_dict['min_num_eng_filters'] = num_filters

                            # set updates
                            energy_optimizer    = Adagrad(lr=sharedX(lr),
                                                          regularizer=Regularizer(l2=lambda_eng))
                            generator_optimizer = Adagrad(lr=sharedX(lr*10.0),
                                                          regularizer=Regularizer(l2=0.0))
                            model_test_name = model_name \
                                              + '_f{}'.format(int(num_filters)) \
                                              + '_h{}'.format(int(hidden_size)) \
                                              + '_e{}'.format(int(expert_size)) \
                                              + '_re{}'.format(int(-np.log10(lambda_eng))) \
                                              + '_lr{}'.format(int(-np.log10(lr))) \

                            train_model(data_stream=data_stream,
                                        energy_optimizer=energy_optimizer,
                                        generator_optimizer=generator_optimizer,
                                        model_config_dict=model_config_dict,
                                        model_test_name=model_test_name)
    else:
        model_test_name = model_name +'_TEST'
        model_config_dict['hidden_size']         = 100
        model_config_dict['expert_size']         = 1024
        model_config_dict['min_num_gen_filters'] = 128
        model_config_dict['min_num_eng_filters'] = 128
        model_config_dict['num_sampling']        = 100
        test_model(model_config_dict=model_config_dict,
                   model_test_name=model_test_name)
