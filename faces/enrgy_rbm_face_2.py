import os
import numpy as np
from time import time
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
from collections import OrderedDict
from lib.activations import Rectify, Tanh, Softplus
from lib.updates import Adagrad, Regularizer, RMSprop, Adam
from lib.inits import Normal, Constant
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.theano_utils import floatX, sharedX
from load import faces

def transform(X):
    return floatX(X)/127.5 - 1.

def inverse_transform(X):
    X = (X+1.)/2.
    return X

model_name  = 'ENERGY_RBM_FACE_TANH'
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

##############################
# SET ACTIVATIONS AND OTHERS #
##############################
relu = Rectify()
tanh = Tanh()
softplus = Softplus()

###################
# SET INITIALIZER #
###################
weight_init = Normal(scale=0.01)
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
    linear_w0    = weight_init((num_hiddens,
                                (num_gen_filters0*init_image_size*init_image_size)),
                               'gen_linear_w0')
    linear_bn_w0 = scale_init((num_gen_filters0*init_image_size*init_image_size),
                              'gen_linear_bn_w0')
    linear_bn_b0 = bias_const((num_gen_filters0*init_image_size*init_image_size),
                              'gen_linear_bn_b0')

    # LAYER 1 (DECONV)
    conv_w1    = weight_init((num_gen_filters0, num_gen_filters1, filter_size, filter_size),
                             'gen_conv_w1')
    conv_bn_w1 = scale_init(num_gen_filters1,
                            'gen_conv_bn_w1')
    conv_bn_b1 = bias_const(num_gen_filters1,
                            'gen_conv_bn_b1')

    # LAYER 2 (DECONV)
    conv_w2    = weight_init((num_gen_filters1, num_gen_filters2, filter_size, filter_size),
                             'gen_conv_w2')
    conv_bn_w2 = scale_init(num_gen_filters2,
                            'gen_conv_bn_w2')
    conv_bn_b2 = bias_const(num_gen_filters2,
                            'gen_conv_bn_b2')

    # LAYER 2 (DECONV)
    conv_w3    = weight_init((num_gen_filters2, num_gen_filters3, filter_size, filter_size),
                             'gen_conv_w3')
    conv_bn_w3 = scale_init(num_gen_filters3,
                            'gen_conv_bn_w3')
    conv_bn_b3 = bias_const(num_gen_filters3,
                            'gen_conv_bn_b3')

    # LAYER 3 (DECONV)
    conv_w4 = weight_init((num_gen_filters3, num_channels, filter_size, filter_size),
                          'gen_conv_w4')
    conv_b4 = bias_zero(num_channels,
                        'gen_conv_b4')

    generator_params = [linear_w0, linear_bn_w0, linear_bn_b0,
                        conv_w1, conv_bn_w1, conv_bn_b1,
                        conv_w2, conv_bn_w2, conv_bn_b2,
                        conv_w3, conv_bn_w3, conv_bn_b3,
                        conv_w4, conv_b4]

    generator_entropy_params = [linear_bn_w0,
                                conv_bn_w1,
                                conv_bn_w2]
    def generator_function(hidden_data, is_train=True):
        # layer 0 (linear)
        h0     = relu(batchnorm(X=T.dot(hidden_data, linear_w0), g=linear_bn_w0, b=linear_bn_b0))
        h0     = h0.reshape((h0.shape[0], num_gen_filters0, init_image_size, init_image_size))
        # layer 1 (deconv)
        h1     = relu(batchnorm(deconv(h0, conv_w1, subsample=(2, 2), border_mode=(2, 2)), g=conv_bn_w1, b=conv_bn_b1))
        # layer 2 (deconv)
        h2     = relu(batchnorm(deconv(h1, conv_w2, subsample=(2, 2), border_mode=(2, 2)), g=conv_bn_w2, b=conv_bn_b2))
        # layer 3 (deconv)
        h3     = relu(batchnorm(deconv(h2, conv_w3, subsample=(2, 2), border_mode=(2, 2)), g=conv_bn_w3, b=conv_bn_b3))
        # layer 4 (deconv)
        output = tanh(deconv(h3, conv_w4, subsample=(2, 2), border_mode=(2, 2))+conv_b4.dimshuffle('x', 0, 'x', 'x'))
        return output

    return [generator_function, generator_params, generator_entropy_params]

######################################
# BUILD ENERGY MODEL (FEATURE_MODEL) #
######################################
def set_energy_model(num_hiddens,
                     min_num_eng_filters):
    min_image_size   = 4
    num_eng_filters0 = min_num_eng_filters*1
    num_eng_filters1 = min_num_eng_filters*2
    num_eng_filters2 = min_num_eng_filters*4
    num_eng_filters3 = min_num_eng_filters*8

    # FEATURE LAYER 0 (DECONV)
    conv_w0   = weight_init((num_eng_filters0, num_channels, filter_size, filter_size),
                            'feat_conv_w0')
    conv_b0   = bias_const(num_eng_filters0,
                           'feat_conv_b0')
    # FEATURE LAYER 1 (DECONV)
    conv_w1   = weight_init((num_eng_filters1, num_eng_filters0, filter_size, filter_size),
                            'feat_conv_w1')
    conv_b1   = bias_const(num_eng_filters1,
                           'feat_conv_b1')
    # FEATURE LAYER 2 (DECONV)
    conv_w2   = weight_init((num_eng_filters2, num_eng_filters1, filter_size, filter_size),
                            'feat_conv_w2')
    conv_b2   = bias_const(num_eng_filters2,
                          'feat_conv_b2')
    # FEATURE LAYER 3 (DECONV)
    conv_w3   = weight_init((num_eng_filters3, num_eng_filters2, filter_size, filter_size),
                            'feat_conv_w3')
    conv_b3   = bias_zero(num_eng_filters3,
                          'feat_conv_b3')

    def feature_function(input_data, is_train=True):
        # layer 0 (conv)
        h0 = relu(dnn_conv(input_data, conv_w0, subsample=(2, 2), border_mode=(2, 2))+conv_b0.dimshuffle('x', 0, 'x', 'x'))
        # layer 1 (conv)
        h1 = relu(dnn_conv(        h0, conv_w1, subsample=(2, 2), border_mode=(2, 2))+conv_b1.dimshuffle('x', 0, 'x', 'x'))
        # layer 2 (conv)
        h2 = relu(dnn_conv(        h1, conv_w2, subsample=(2, 2), border_mode=(2, 2))+conv_b2.dimshuffle('x', 0, 'x', 'x'))
        # layer 3 (conv)
        h3 = tanh(dnn_conv(h2, conv_w3, subsample=(2, 2), border_mode=(2, 2))+conv_b3.dimshuffle('x', 0, 'x', 'x'))
        f  = T.flatten(h3, 2)
        return f

    # ENERGY LAYER (LINEAR)
    # feature_mean = bias_zero((num_eng_filters3*(min_image_size*min_image_size), ),
    #                          'feature_mean')
    # feature_std  = bias_zero((num_eng_filters3*(min_image_size*min_image_size), ),
    #                          'feature_std')
    linear_w0    = weight_init((num_eng_filters3*(min_image_size*min_image_size),
                                num_hiddens),
                               'eng_linear_w0')
    linear_b0    = bias_zero(num_hiddens,
                             'eng_linear_b0')

    energy_params = [conv_w0, conv_b0,
                     conv_w1, conv_b1,
                     conv_w2, conv_b2,
                     conv_w3, conv_b3,
                     # feature_mean, feature_std,
                     linear_w0, linear_b0]

    def energy_function(feature_data, is_train=True):
        # feature-wise std
        # feature_std_inv = T.inv(T.nnet.softplus(feature_std)+1e-10)
        feature_std_inv = 1.0
        # energy hidden-feature
        e = softplus(T.dot(feature_data*feature_std_inv, linear_w0)+linear_b0)
        e = T.sum(-e, axis=1)
        # energy feature prior
        # e += 0.5*T.sum(T.sqr(feature_std_inv)*T.sqr(feature_data), axis=1)
        return e

    return [feature_function, energy_function, energy_params]

######################
# FULL MODEL UPDATER #
######################
def set_updater_function(feature_function,
                         energy_function,
                         generator_function,
                         energy_params,
                         generator_params,
                         generator_entropy_params,
                         energy_optimizer,
                         generator_optimizer):

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
    whole_data    = T.concatenate([input_data, sample_data], axis=0)
    whole_feature = feature_function(whole_data, is_train=True)
    input_feature  = whole_feature[:input_data.shape[0]]
    sample_feature = whole_feature[input_data.shape[0]:]
    # get energy value
    input_energy  = energy_function(input_feature, is_train=True)
    sample_energy = energy_function(sample_feature, is_train=True)

    # get energy function cost (positive, negative)
    positive_phase      = T.mean(input_energy*annealing_scale)
    negative_phase      = T.mean(sample_energy*annealing_scale)

    # get cost for energy/generator
    energy_updates_cost    = positive_phase - negative_phase
    generator_updates_cost = negative_phase

    # get energy updates
    energy_updates = energy_optimizer(energy_params, energy_updates_cost)

    # get generator updates
    generator_updates = generator_optimizer(generator_params, generator_updates_cost)

    # updater function input
    updater_function_inputs  = [input_data,
                                hidden_data,
                                annealing]

    # updater function output
    updater_function_outputs = [input_energy,
                                sample_energy]

    # updater function
    updater_function = theano.function(inputs=updater_function_inputs,
                                       outputs=updater_function_outputs,
                                       updates=energy_updates+generator_updates,
                                       on_unused_input='ignore')
    return updater_function

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
                                  generator_entropy_params,
                                  generator_optimizer):

    # set input data, hidden data, noise_data annealing rate
    input_data  = T.tensor4(name='input_data',
                            dtype=theano.config.floatX)
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
    # sample_data = T.clip(sample_data+noise_data, -1., 1.)

    # get feature data
    input_feature  = feature_function(input_data, is_train=True)
    sample_feature = feature_function(sample_data, is_train=True)

    # get energy value
    input_energy = energy_function(input_feature, is_train=True)
    sample_energy = energy_function(sample_feature, is_train=True)

    # entropy regularizer
    entropy_cost = 0.0
    for entropy_param in generator_params:
        entropy_cost += T.sum(T.exp(-T.abs_(entropy_param)))

    # get generator update cost
    generator_updates_cost = T.mean(sample_energy*annealing_scale)

    # get generator updates
    generator_updates = generator_optimizer(generator_params, generator_updates_cost)

    # update function input
    update_function_inputs  = [input_data,
                               hidden_data,
                               noise_data,
                               annealing]

    # update function output
    update_function_outputs = [input_energy,
                               sample_energy,
                               entropy_cost]

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
def set_sampling_function(generator_function):

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
def train_model(data_stream,
                energy_optimizer,
                generator_optimizer,
                model_config_dict,
                model_test_name):

    [generator_function, generator_params, generator_entropy_params] = set_generator_model(model_config_dict['hidden_size'],
                                                                                           model_config_dict['min_num_gen_filters'])
    [feature_function, energy_function, energy_params] = set_energy_model(model_config_dict['hidden_size'],
                                                                          model_config_dict['min_num_eng_filters'])
    # compile functions
    print 'COMPILING ENERGY UPDATER'
    t=time()
    updater_function = set_updater_function(feature_function=feature_function,
                                            energy_function=energy_function,
                                            generator_function=generator_function,
                                            energy_params=energy_params,
                                            generator_params=generator_params,
                                            generator_entropy_params=generator_entropy_params,
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
    #                                                   generator_optimizer=generator_optimizer)
    # print '%.2f SEC '%(time()-t)
    # print 'COMPILING EVALUATION FUNCTION'
    # t=time()
    # evaluation_function = set_evaluation_and_sampling_function(feature_function=feature_function,
    #                                                            energy_function=energy_function,
    #                                                            generator_function=generator_function)
    # print '%.2f SEC '%(time()-t)
    print 'COMPILING SAMPLING FUNCTION'
    t=time()
    sampling_function = set_sampling_function(generator_function=generator_function)
    print '%.2f SEC '%(time()-t)

    # set fixed hidden data for sampling
    fixed_hidden_data  = floatX(np_rng.uniform(low=-model_config_dict['hidden_distribution'],
                                               high=model_config_dict['hidden_distribution'],
                                               size=(model_config_dict['num_display'], model_config_dict['hidden_size'])))
    fixed_hidden_data  = np.arctanh(fixed_hidden_data)

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
            hidden_data  = np.arctanh(hidden_data)

            # update generator
            # generator_update_inputs = [input_data,
            #                            hidden_data,
            #                            noise_data,
            #                            e]
            # [input_energy_val, sample_energy_val, entropy_cost] = generator_updater(*generator_update_inputs)

            # update energy function
            energy_update_inputs = [input_data,
                                    hidden_data,
                                    e]
            [input_energy_val, sample_energy_val, ] = updater_function(*energy_update_inputs)

            # get output values
            input_energy  = input_energy_val.mean()
            sample_energy = sample_energy_val.mean()

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
                print '     input energy     : ', input_energy
                print '----------------------------------------------------------------'
                print '     sample energy    : ', sample_energy
                print '================================================================'

            if batch_count%100==0:
                # sample data
                save_as = samples_dir + '/' + model_test_name + '_SAMPLES{}.png'.format(batch_count)
                sample_data = sampling_function(fixed_hidden_data)[0]
                sample_data = np.asarray(sample_data)
                color_grid_vis(inverse_transform(sample_data).transpose([0,2,3,1]), (16, 16), save_as)
                np.save(file=samples_dir + '/' + model_test_name +'_input_energy',
                        arr=np.asarray(input_energy_list))
                np.save(file=samples_dir + '/' + model_test_name +'_sample_energy',
                        arr=np.asarray(sample_energy_list))

if __name__=="__main__":

    model_config_dict = OrderedDict()
    model_config_dict['batch_size']          = 64
    model_config_dict['num_display']         = 16*16
    model_config_dict['hidden_distribution'] = 1.
    model_config_dict['epochs']              = 200

    #################
    # LOAD DATA SET #
    #################
    _ , data_stream = faces(batch_size=model_config_dict['batch_size'])

    hidden_size_list = [100]
    num_filters_list = [32]
    lr_list          = [1e-5]
    dropout_list     = [False,]
    lambda_eng_list  = [1e-5]
    lambda_gen_list  = [1e-5]
    init_noise_list  = [1e-2]
    noise_decay_list = [0.98]

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
                                    energy_optimizer    = RMSprop(lr=sharedX(lr),
                                                                  regularizer=Regularizer(l2=lambda_eng))
                                    generator_optimizer = RMSprop(lr=sharedX(lr*10.0),
                                                                  regularizer=Regularizer(l2=lambda_gen))
                                    model_test_name = model_name \
                                                      + '_f{}'.format(int(num_filters)) \
                                                      + '_h{}'.format(int(hidden_size)) \
                                                      + '_d{}'.format(int(dropout)) \
                                                      + '_re{}'.format(int(-np.log10(lambda_eng))) \
                                                      + '_rg{}'.format(int(-np.log10(lambda_gen))) \
                                                      + '_n{}'.format(int(-np.log10(init_noise))) \
                                                      + '_d{}'.format(int(1 if noise_decay is 1.0 else 0)) \
                                                      + '_lr{}'.format(int(-np.log10(lr))) \

                                    train_model(data_stream=data_stream,
                                                energy_optimizer=energy_optimizer,
                                                generator_optimizer=generator_optimizer,
                                                model_config_dict=model_config_dict,
                                                model_test_name=model_test_name)
