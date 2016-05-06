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
from load import faces
from lib.save_utils import save_model

def transform(X):
    return floatX(X)/127.5 - 1.

def inverse_transform(X):
    X = (X+1.)/2.
    return X

model_name  = 'MOMENT_NORMAL_AE_FACE'
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
lrelu = LeakyRectify()
relu  = Rectify()
tanh  = Tanh()
softplus = Softplus()

###################
# SET INITIALIZER #
###################
weight_init = Normal(scale=0.01)
scale_init  = Constant(c=1.0)
bias_zero   = Constant(c=0.0)

#################
# BUILD DECODER #
#################
def set_decoder_model(num_hiddens,
                      min_num_gen_filters):
    # initial square image size
    init_image_size  = 4
    
    # set num of filters for each layer
    num_gen_filters0 = min_num_gen_filters*8
    num_gen_filters1 = min_num_gen_filters*4
    num_gen_filters2 = min_num_gen_filters*2
    num_gen_filters3 = min_num_gen_filters*1

    # LAYER 0 (LINEAR W/ BN)
    linear_w0 = weight_init((num_hiddens,
                             (num_gen_filters0*init_image_size*init_image_size)),
                            'dec_linear_w0')
    linear_b0 = bias_zero((num_gen_filters0*init_image_size*init_image_size),
                          'dec_linear_b0')

    # LAYER 1 (DECONV)
    conv_w1 = weight_init((num_gen_filters0, num_gen_filters1) + filter_shape,
                          'dec_conv_w1')
    conv_b1 = bias_zero(num_gen_filters1,
                        'dec_conv_b1')

    # LAYER 2 (DECONV)
    conv_w2 = weight_init((num_gen_filters1, num_gen_filters2) + filter_shape,
                          'dec_conv_w2')
    conv_b2 = bias_zero(num_gen_filters2,
                        'dec_conv_b2')

    # LAYER 3 (DECONV)
    conv_w3 = weight_init((num_gen_filters2, num_gen_filters3) + filter_shape,
                          'dec_conv_w3')
    conv_b3 = bias_zero(num_gen_filters3,
                        'dec_conv_b3')

    # LAYER OUTPUT (DECONV)
    conv_w4 = weight_init((num_gen_filters3, num_channels) + filter_shape,
                          'dec_conv_w4')
    conv_b4 = bias_zero(num_channels,
                        'dec_conv_b4')

    decoder_params = [linear_w0, linear_b0,
                      conv_w1, conv_b1,
                      conv_w2, conv_b2,
                      conv_w3, conv_b3,
                      conv_w4, conv_b4]


    def decoder_function(hidden_data):
        # layer 0 (linear)
        h0 = T.dot(hidden_data, linear_w0) + linear_b0
        h0 = h0.reshape((h0.shape[0], num_gen_filters0, init_image_size, init_image_size))
        # layer 1 (deconv)
        h1 = deconv(relu(h0), conv_w1, subsample=(2, 2), border_mode=(2, 2)) + conv_b1.dimshuffle('x', 0, 'x', 'x')
        # layer 2 (deconv)
        h2 = deconv(relu(h1), conv_w2, subsample=(2, 2), border_mode=(2, 2)) + conv_b2.dimshuffle('x', 0, 'x', 'x')
        # layer 3 (deconv)
        h3 = deconv(relu(h2), conv_w3, subsample=(2, 2), border_mode=(2, 2)) + conv_b3.dimshuffle('x', 0, 'x', 'x')
        # layer_output (deconv)
        h4 = deconv(relu(h3), conv_w4, subsample=(2, 2), border_mode=(2, 2)) + conv_b4.dimshuffle('x', 0, 'x', 'x')
        output = tanh(h4)
        return [[hidden_data,
                 T.flatten(h0,2),
                 T.flatten(h1,2),
                 T.flatten(h2,2),
                 T.flatten(h3,2),
                 T.flatten(h4,2)],
                output]


    return [decoder_function,
            decoder_params]

#######################
# BUILD ENCODER MODEL #
#######################
def set_encoder_model(num_hiddens,
                      min_num_eng_filters):

    # minimum square image size
    min_image_size   = 4

    # set num of filters for each layer
    num_eng_filters0 = min_num_eng_filters*1
    num_eng_filters1 = min_num_eng_filters*2
    num_eng_filters2 = min_num_eng_filters*4
    num_eng_filters3 = min_num_eng_filters*8

    # FEATURE LAYER 0 (CONV)
    conv_w0 = weight_init((num_eng_filters0, num_channels) + filter_shape,
                          'enc_conv_w0')
    bn_w0   = scale_init(num_eng_filters0,
                         'enc_bn_w0')
    bn_b0   = bias_zero(num_eng_filters0,
                        'enc_bn_b0')
    # FEATURE LAYER 1 (CONV)
    conv_w1 = weight_init((num_eng_filters1, num_eng_filters0) + filter_shape,
                          'enc_conv_w1')
    bn_w1   = scale_init(num_eng_filters1,
                         'enc_bn_w1')
    bn_b1   = bias_zero(num_eng_filters1,
                        'enc_bn_b1')
    # FEATURE LAYER 2 (CONV)
    conv_w2 = weight_init((num_eng_filters2, num_eng_filters1) + filter_shape,
                          'enc_conv_w2')
    bn_w2   = scale_init(num_eng_filters2,
                         'enc_bn_w2')
    bn_b2   = bias_zero(num_eng_filters2,
                        'enc_bn_b2')
    # FEATURE LAYER 3 (CONV)
    conv_w3 = weight_init((num_eng_filters3, num_eng_filters2) + filter_shape,
                          'enc_conv_w3')
    bn_w3   = scale_init(num_eng_filters3,
                         'enc_bn_w3')
    bn_b3   = bias_zero(num_eng_filters3,
                        'enc_bn_b3')

    # ENCODER HIDDEN
    hidden_w = weight_init((num_eng_filters3*(min_image_size*min_image_size),
                            num_hiddens),
                           'hidden_w')
    hidden_b = bias_zero(num_hiddens,
                         'hidden_b')

    encoder_params = [conv_w0, bn_w0, bn_b0,
                      conv_w1, bn_w1, bn_b1,
                      conv_w2, bn_w2, bn_b2,
                      conv_w3, bn_w3, bn_b3,
                      hidden_w, hidden_b]

    def encoder_function(input_data):
        # layer 0 (conv)
        h0 = dnn_conv(input_data, conv_w0, subsample=(2, 2), border_mode=(2, 2))
        h0 = relu(batchnorm(h0, g=bn_w0, b=bn_b0))
        # layer 1 (conv)
        h1 = dnn_conv(        h0, conv_w1, subsample=(2, 2), border_mode=(2, 2))
        h1 = relu(batchnorm(h1, g=bn_w1, b=bn_b1))
        # layer 2 (conv)
        h2 = dnn_conv(        h1, conv_w2, subsample=(2, 2), border_mode=(2, 2))
        h2 = relu(batchnorm(h2, g=bn_w2, b=bn_b2))
        # layer 3 (conv)
        h3 = dnn_conv(        h2, conv_w3, subsample=(2, 2), border_mode=(2, 2))
        h3 = T.flatten(relu(batchnorm(h3, g=bn_w3, b=bn_b3)), 2)
        # layer output
        hidden_data = T.dot(h3, hidden_w)+hidden_b
        return hidden_data

    return [encoder_function,
            encoder_params]

################
#MODEL UPDATER #
################
def set_updater_function(encoder_function,
                         decoder_function,
                         encoder_params,
                         decoder_params,
                         optimizer):

    # positive visible data
    positive_visible_data = T.tensor4(name='positive_visible_data',
                                      dtype=theano.config.floatX)
    # negative hidden data
    negative_hidden_data = T.matrix(name='negative_hidden_data',
                                    dtype=theano.config.floatX)
    # moment weight
    moment_cost_weight = T.scalar(name='moment_cost_weight',
                                  dtype=theano.config.floatX)

    ##################
    # positive phase #
    ##################
    # positive encoder
    positive_encoder_hidden  = encoder_function(positive_visible_data)
    # positive decoder
    positive_decoder_data    = decoder_function(positive_encoder_hidden)
    positive_decoder_hiddens = positive_decoder_data[0]
    positive_decoder_output  = positive_decoder_data[1]

    ##################
    # negative phase #
    ##################
    # negative decoder
    negative_decoder_outputs = decoder_function(negative_hidden_data)
    negative_decoder_hiddens = negative_decoder_outputs[0]
    negative_decoder_output  = negative_decoder_outputs[1]

    # positive reconstruction cost
    positive_recon_cost = T.sum(T.sqr(positive_visible_data-positive_decoder_output), axis=(1,2,3))

    # moment matching cost
    moment_match_cost = 0
    for i in xrange(len(positive_decoder_hiddens)):
        pos_feat = positive_decoder_hiddens[i]
        neg_feat = negative_decoder_hiddens[i]
        moment_match_cost += T.mean(T.sqr(T.mean(pos_feat, axis=0)-T.mean(neg_feat, axis=0)))
        moment_match_cost += T.mean(T.sqr(T.mean(T.sqr(pos_feat), axis=0)-T.mean(T.sqr(neg_feat), axis=0)))

    model_updater_cost = T.mean(positive_recon_cost) + moment_cost_weight*T.mean(moment_match_cost)
    model_updater_dict = optimizer(encoder_params+decoder_params,
                                   model_updater_cost)

    model_updater_inputs = [positive_visible_data,
                            negative_hidden_data,
                            moment_cost_weight]
    model_updater_outputs = [positive_recon_cost,
                             moment_match_cost,
                             model_updater_cost]


    model_updater_function = theano.function(inputs=model_updater_inputs,
                                             outputs=model_updater_outputs,
                                             updates=model_updater_dict,
                                             on_unused_input='ignore')
    return model_updater_function

###########
# SAMPLER #
###########
def set_sampling_function(decoder_function):

    hidden_data = T.matrix(name='hidden_data',
                           dtype=theano.config.floatX)

    # decoder
    decoder_outputs = decoder_function(hidden_data)
    decoder_samples = decoder_outputs[1]

    function_inputs = [hidden_data,]
    function_outputs = [decoder_samples,]

    function = theano.function(inputs=function_inputs,
                               outputs=function_outputs,
                               on_unused_input='ignore')
    return function

###########
# TRAINER #
###########
def train_model(data_stream,
                model_optimizer,
                model_config_dict,
                model_test_name):

    encoder_model = set_encoder_model(model_config_dict['hidden_size'],
                                      model_config_dict['min_num_gen_filters'])
    encoder_function   = encoder_model[0]
    encoder_parameters = encoder_model[1]
    decoder_model = set_decoder_model(model_config_dict['hidden_size'],
                                      model_config_dict['min_num_eng_filters'])
    decoder_function   = decoder_model[0]
    decoder_parameters = decoder_model[1]

    # compile functions
    print 'COMPILING UPDATER FUNCTION'
    t=time()
    updater_function = set_updater_function(encoder_function=encoder_function,
                                            decoder_function=decoder_function,
                                            encoder_params=encoder_parameters,
                                            decoder_params=decoder_parameters,
                                            optimizer=model_optimizer)
    print '%.2f SEC '%(time()-t)
    print 'COMPILING SAMPLING FUNCTION'
    t=time()
    sampling_function = set_sampling_function(decoder_function=decoder_function)
    print '%.2f SEC '%(time()-t)

    # set fixed hidden data for sampling
    fixed_hidden_data  = floatX(np_rng.normal(size=(model_config_dict['num_display'], model_config_dict['hidden_size'])))

    print 'START TRAINING'
    # for each epoch
    recon_cost_list        = []
    moment_match_cost_list = []
    model_cost_list        = []
    batch_count = 0
    for e in xrange(model_config_dict['epochs']):
        # train phase
        batch_iters = data_stream.get_epoch_iterator()
        # for each batch
        for b, batch_data in enumerate(batch_iters):
            # set update function inputs
            positive_visible_data = transform(batch_data[0])
            negative_hidden_data  = floatX(np_rng.normal(size=(positive_visible_data.shape[0], model_config_dict['hidden_size'])))
            moment_cost_weight    = 1.0

            updater_inputs = [positive_visible_data,
                              negative_hidden_data,
                              moment_cost_weight]
            updater_outputs = updater_function(*updater_inputs)

            recon_cost        = updater_outputs[0].mean()
            moment_match_cost = updater_outputs[1].mean()
            model_cost        = updater_outputs[2].mean()
            recon_cost_list.append(recon_cost)
            moment_match_cost_list.append(moment_match_cost)
            model_cost_list.append(model_cost)
            # batch count up
            batch_count += 1

            if batch_count%1==0:
                print '================================================================'
                print 'BATCH ITER #{}'.format(batch_count), model_test_name
                print '================================================================'
                print '   TRAIN RESULTS'
                print '================================================================'
                print '     recon  cost : ', recon_cost_list[-1]
                print '----------------------------------------------------------------'
                print '     moment cost : ', moment_match_cost_list[-1]
                print '----------------------------------------------------------------'
                print '     model  cost : ', model_cost_list[-1]
                print '================================================================'

            if batch_count%100==0:
                # sample data
                sample_data = sampling_function(fixed_hidden_data)[0]
                save_as = samples_dir + '/' + model_test_name + '_SAMPLES(NEGATIVE){}.png'.format(batch_count)
                color_grid_vis(inverse_transform(sample_data).transpose([0,2,3,1]), (16, 16), save_as)

                # save costs
                np.save(file=samples_dir + '/' + model_test_name +'_recon_cost',
                        arr=np.asarray(recon_cost_list))
                np.save(file=samples_dir + '/' + model_test_name +'_moment_cost',
                        arr=np.asarray(moment_match_cost_list))
                np.save(file=samples_dir + '/' + model_test_name +'_model_cost',
                        arr=np.asarray(model_cost_list))
            if batch_count%1000==0:
                save_as = samples_dir + '/' + model_test_name + '_MODEL.pkl'
                save_model(tensor_params_list=decoder_parameters,
                           save_to=save_as)


if __name__=="__main__":

    model_config_dict = OrderedDict()
    model_config_dict['batch_size']          = 12*12
    model_config_dict['num_display']         = 16*16
    model_config_dict['hidden_distribution'] = 1.
    model_config_dict['epochs']              = 200

    #################
    # LOAD DATA SET #
    #################
    _ , data_stream = faces(batch_size=model_config_dict['batch_size'])

    hidden_size_list = [100]
    num_filters_list = [128]
    lr_list          = [1e-3]
    dropout_list     = [False,]
    lambda_eng_list  = [1e-10]
    lambda_gen_list  = [1e-10]

    for lr in lr_list:
        for num_filters in num_filters_list:
            for hidden_size in hidden_size_list:
                for lambda_eng in lambda_eng_list:
                    for lambda_gen in lambda_gen_list:
                        model_config_dict['hidden_size']         = hidden_size
                        model_config_dict['min_num_gen_filters'] = num_filters
                        model_config_dict['min_num_eng_filters'] = num_filters

                        # set updates
                        model_optimizer = Adagrad(lr=sharedX(lr),
                                                  regularizer=Regularizer(l2=lambda_eng))
                        model_test_name = model_name \
                                          + '_f{}'.format(int(num_filters)) \
                                          + '_h{}'.format(int(hidden_size)) \
                                          + '_re{}'.format(int(-np.log10(lambda_eng))) \
                                          + '_rg{}'.format(int(-np.log10(lambda_gen))) \
                                          + '_lr{}'.format(int(-np.log10(lr))) \

                        train_model(data_stream=data_stream,
                                    model_optimizer=model_optimizer,
                                    model_config_dict=model_config_dict,
                                    model_test_name=model_test_name)
