import os
import theano
import theano.tensor as T
import numpy as np
import h5py
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool
from lib.theano_utils import floatX, sharedX
from lib.ops import deconv
from lib.activations import Rectify, Tanh
from lib.inits import Normal, Constant
from lib.rng import np_rng
from lib.vis import color_grid_vis
from lib.updates import Adagrad, Regularizer, RMSprop, Adam
from load import faces
from time import time


vgg_filepath = '/data/lisatmp4/taesup/data/vgg/vgg16_weights.h5'

model_name  = 'vgg_moment_match_face'
samples_dir = 'samples/%s'%model_name
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

relu        = Rectify()
tanh        = Tanh()
weight_init = Normal(scale=0.01)
bias_init   = Constant(c=0.0)

def transform(X):
    return floatX(X)/127.5 - 1.

def inverse_transform(X):
    X = (X+1.)/2.
    return X

def load_vgg_feature_extractor():
    vgg_param_dict = h5py.File(vgg_filepath, 'r')
    # input_channel x output_channel x filter_size x filter_size
    # conv stage 0 (64x64=>32x32)
    # (3 x 64 x 3 x 3)
    conv_w0_0 = sharedX(vgg_param_dict['layer_1']['param_0'], name='feat_conv_w0_0')
    conv_b0_0 = sharedX(vgg_param_dict['layer_1']['param_1'], name='feat_conv_b0_0')

    # (64 x 64 x 3 x 3)
    conv_w0_1 = sharedX(vgg_param_dict['layer_3']['param_0'], name='feat_conv_w0_1')
    conv_b0_1 = sharedX(vgg_param_dict['layer_3']['param_1'], name='feat_conv_b0_1')

    # conv stage 1 (32x32=>16x16)
    # (64 x 128 x 3 x 3)
    conv_w1_0 = sharedX(vgg_param_dict['layer_6']['param_0'], name='feat_conv_w1_0')
    conv_b1_0 = sharedX(vgg_param_dict['layer_6']['param_1'], name='feat_conv_w1_0')

    # (128 x 128 x 3 x 3)
    conv_w1_1 = sharedX(vgg_param_dict['layer_8']['param_0'], name='feat_conv_w1_1')
    conv_b1_1 = sharedX(vgg_param_dict['layer_8']['param_1'], name='feat_conv_b1_1')

    # conv stage 2 (16x16=>8x8)
    # (128 x 256 x 3 x 3)
    conv_w2_0 = sharedX(vgg_param_dict['layer_11']['param_0'], name='feat_conv_w2_0')
    conv_b2_0 = sharedX(vgg_param_dict['layer_11']['param_1'], name='feat_conv_b2_0')

    # (256 x 256 x 3 x 3)
    conv_w2_1 = sharedX(vgg_param_dict['layer_13']['param_0'], name='feat_conv_w2_1')
    conv_b2_1 = sharedX(vgg_param_dict['layer_13']['param_1'], name='feat_conv_b2_1')

    # (256 x 256 x 3 x 3)
    conv_w2_2 = sharedX(vgg_param_dict['layer_15']['param_0'], name='feat_conv_w2_2')
    conv_b2_2 = sharedX(vgg_param_dict['layer_15']['param_1'], name='feat_conv_b2_2')

    # conv stage 3 (8x8=>4x4)
    # (256 x 512 x 3 x 3)
    conv_w3_0 = sharedX(vgg_param_dict['layer_18']['param_0'], name='feat_conv_w3_0')
    conv_b3_0 = sharedX(vgg_param_dict['layer_18']['param_1'], name='feat_conv_b3_0')

    # (512 x 512 x 3 x 3)
    conv_w3_1 = sharedX(vgg_param_dict['layer_20']['param_0'], name='feat_conv_w3_1')
    conv_b3_1 = sharedX(vgg_param_dict['layer_20']['param_1'], name='feat_conv_b3_1')

    # (512 x 512 x 3 x 3)
    conv_w3_2 = sharedX(vgg_param_dict['layer_22']['param_0'], name='feat_conv_w3_2')
    conv_b3_2 = sharedX(vgg_param_dict['layer_22']['param_1'], name='feat_conv_b3_2')

    # conv stage 4 (4x4=>2x2)
    # (512 x 512 x 3 x 3)
    conv_w4_0 = sharedX(vgg_param_dict['layer_25']['param_0'], name='feat_conv_w4_0')
    conv_b4_0 = sharedX(vgg_param_dict['layer_25']['param_1'], name='feat_conv_b4_0')

    # (512 x 512 x 3 x 3)
    conv_w4_1 = sharedX(vgg_param_dict['layer_27']['param_0'], name='feat_conv_w4_1')
    conv_b4_1 = sharedX(vgg_param_dict['layer_27']['param_1'], name='feat_conv_b4_1')

    # (512 x 512 x 3 x 3)
    conv_w4_2 = sharedX(vgg_param_dict['layer_29']['param_0'], name='feat_conv_w4_2')
    conv_b4_2 = sharedX(vgg_param_dict['layer_29']['param_1'], name='feat_conv_b4_2')

    # parameter_set = [conv_w0_0, conv_b0_0, conv_w0_1, conv_b0_1,
    #                  conv_w1_0, conv_b1_0, conv_w1_1, conv_b1_1,
    #                  conv_w2_0, conv_b2_0, conv_w2_1, conv_b2_1, conv_w2_2, conv_b2_2,
    #                  conv_w3_0, conv_b3_0, conv_w3_1, conv_b3_1, conv_w3_2, conv_b3_2,
    #                  conv_w4_0, conv_b4_0, conv_w4_1, conv_b4_1, conv_w4_2, conv_b4_2]

    def feature_extractor(input_data):
        # conv stage 0 (64x64=>32x32)
        h0_0 = relu(dnn_conv(input_data, conv_w0_0, border_mode=(1, 1))+conv_b0_0.dimshuffle('x', 0, 'x', 'x'))
        h0_1 = relu(dnn_conv(      h0_0, conv_w0_1, border_mode=(1, 1))+conv_b0_1.dimshuffle('x', 0, 'x', 'x'))
        h0   =      dnn_pool(      h0_1, ws=(2, 2), stride=(2, 2))
        # conv stage 1 (32x32=>16x16)
        h1_0 = relu(dnn_conv(        h0, conv_w1_0, border_mode=(1, 1))+conv_b1_0.dimshuffle('x', 0, 'x', 'x'))
        h1_1 = relu(dnn_conv(      h1_0, conv_w1_1, border_mode=(1, 1))+conv_b1_1.dimshuffle('x', 0, 'x', 'x'))
        h1   =      dnn_pool(      h1_1, ws=(2, 2), stride=(2, 2))
        # conv stage 2 (16x16=>8x8)
        h2_0 = relu(dnn_conv(        h1, conv_w2_0, border_mode=(1, 1))+conv_b2_0.dimshuffle('x', 0, 'x', 'x'))
        h2_1 = relu(dnn_conv(      h2_0, conv_w2_1, border_mode=(1, 1))+conv_b2_1.dimshuffle('x', 0, 'x', 'x'))
        h2_2 = relu(dnn_conv(      h2_1, conv_w2_2, border_mode=(1, 1))+conv_b2_2.dimshuffle('x', 0, 'x', 'x'))
        h2   =      dnn_pool(      h2_2, ws=(2, 2), stride=(2, 2))
        # conv stage 3 (8x8=>4x4)
        h3_0 = relu(dnn_conv(        h2, conv_w3_0, border_mode=(1, 1))+conv_b3_0.dimshuffle('x', 0, 'x', 'x'))
        h3_1 = relu(dnn_conv(      h3_0, conv_w3_1, border_mode=(1, 1))+conv_b3_1.dimshuffle('x', 0, 'x', 'x'))
        h3_2 = relu(dnn_conv(      h3_1, conv_w3_2, border_mode=(1, 1))+conv_b3_2.dimshuffle('x', 0, 'x', 'x'))
        h3   =      dnn_pool(      h3_2, ws=(2, 2), stride=(2, 2))
        # conv stage 4 (4x4=>2x2)
        h4_0 = relu(dnn_conv(        h3, conv_w4_0, border_mode=(1, 1))+conv_b4_0.dimshuffle('x', 0, 'x', 'x'))
        h4_1 = relu(dnn_conv(      h4_0, conv_w4_1, border_mode=(1, 1))+conv_b4_1.dimshuffle('x', 0, 'x', 'x'))
        h4_2 = relu(dnn_conv(      h4_1, conv_w4_2, border_mode=(1, 1))+conv_b4_2.dimshuffle('x', 0, 'x', 'x'))
        h4   =      dnn_pool(      h4_2, ws=(2, 2), stride=(2, 2))

        return [T.flatten(h0_0, 2), T.flatten(h0_1, 2),
                T.flatten(h1_0, 2), T.flatten(h1_1, 2),
                T.flatten(h2_0, 2), T.flatten(h2_1, 2), T.flatten(h2_2, 2),
                T.flatten(h3_0, 2), T.flatten(h3_1, 2), T.flatten(h3_2, 2),
                T.flatten(h4_0, 2), T.flatten(h4_1, 2), T.flatten(h4_2, 2),
                T.flatten(  h4, 2)]

    return feature_extractor


def set_generator_model(num_hiddens):

    init_image_size  = 2
    filter_shape     = (3, 3)
    num_gen_filters0 = 512
    num_gen_filters1 = 512
    num_gen_filters2 = 256
    num_gen_filters3 = 128
    num_gen_filters4 =  64
    num_gen_filters5 =   3

    # linear layer (hidden_size, 2x2x512)
    linear_w0 = weight_init((num_hiddens, (num_gen_filters0*init_image_size*init_image_size)),
                            'gen_linear_w0')
    linear_b0 = bias_init((num_gen_filters0*init_image_size*init_image_size),
                          'gen_linear_b0')

    # conv stage 0 (2x2x512, 4x4x512)
    conv_w0_0 = weight_init((num_gen_filters0, num_gen_filters0) + filter_shape,
                            'gen_conv_w0_0') # 2x2x512=>4x4x512
    conv_b0_0 = bias_init(num_gen_filters0,
                          'gen_conv_b0_0')

    conv_w0_1 = weight_init((num_gen_filters0, num_gen_filters0) + filter_shape,
                            'gen_conv_w0_1') # 4x4x512=>4x4x512
    conv_b0_1 = bias_init(num_gen_filters0,
                          'gen_conv_b0_1')

    conv_w0_2 = weight_init((num_gen_filters0, num_gen_filters0) + filter_shape,
                            'gen_conv_w0_2') # 4x4x512=>4x4x512
    conv_b0_2 = bias_init(num_gen_filters0,
                          'gen_conv_b0_2')

    # conv stage 1 (4x4x512, 8x8x512)
    conv_w1_0 = weight_init((num_gen_filters0, num_gen_filters1) + filter_shape,
                            'gen_conv_w1_0') # 4x4x512=>8x8x512
    conv_b1_0 = bias_init(num_gen_filters1,
                          'gen_conv_b1_0')

    conv_w1_1 = weight_init((num_gen_filters1, num_gen_filters1) + filter_shape,
                            'gen_conv_w1_1') # 8x8x512=>8x8x512
    conv_b1_1 = bias_init(num_gen_filters1,
                          'gen_conv_b1_1')

    conv_w1_2 = weight_init((num_gen_filters1, num_gen_filters1) + filter_shape,
                            'gen_conv_w1_2') # 8x8x512=>8x8x512
    conv_b1_2 = bias_init(num_gen_filters1,
                          'gen_conv_b1_2')

    # conv stage 2 (8x8x512, 16x16x256)
    conv_w2_0 = weight_init((num_gen_filters1, num_gen_filters2) + filter_shape,
                            'gen_conv_w2_0') # 8x8x512=>16x16x256
    conv_b2_0 = bias_init(num_gen_filters2,
                          'gen_conv_b2_0')

    conv_w2_1 = weight_init((num_gen_filters2, num_gen_filters2) + filter_shape,
                            'gen_conv_w2_1') # 16x16x256=>16x16x256
    conv_b2_1 = bias_init(num_gen_filters2,
                          'gen_conv_b2_1')

    conv_w2_2 = weight_init((num_gen_filters2, num_gen_filters2) + filter_shape,
                            'gen_conv_w2_2') # 16x16x256=>16x16x256
    conv_b2_2 = bias_init(num_gen_filters2,
                          'gen_conv_b2_2')

    # conv stage 3 (16x16x256, 32x32x128)
    conv_w3_0 = weight_init((num_gen_filters2, num_gen_filters3) + filter_shape,
                            'gen_conv_w3_0') #16x16x256=>32x32x128
    conv_b3_0 = bias_init(num_gen_filters3,
                          'gen_conv_b3_0')

    conv_w3_1 = weight_init((num_gen_filters3, num_gen_filters3) + filter_shape,
                            'gen_conv_w3_1') #32x32x128=>32x32x128
    conv_b3_1 = bias_init(num_gen_filters3,
                          'gen_conv_b3_1')

    # conv stage 4 (32x32x128, 64x64x64)
    conv_w4_0 = weight_init((num_gen_filters3, num_gen_filters4) + filter_shape,
                            'gen_conv_w4_0') # 32x32x128=>64x64x64
    conv_b4_0 = bias_init(num_gen_filters4,
                          'gen_conv_b4_0')

    conv_w4_1 = weight_init((num_gen_filters4, num_gen_filters4) + filter_shape,
                            'gen_conv_w4_1') # 64x64x64=>64x64x64
    conv_b4_1 = bias_init(num_gen_filters4,
                          'gen_conv_b4_1')

    # output (32x32x128, 64x64x64)
    conv_w5   = weight_init((num_gen_filters5, num_gen_filters4) + filter_shape,
                            'gen_conv_w5')
    conv_b5   = bias_init(num_gen_filters5,
                          'gen_conv_b5')

    parameter_set = [linear_w0, linear_b0,
                     conv_w0_0, conv_b0_0, conv_w0_1, conv_b0_1, conv_w0_2, conv_b0_2,
                     conv_w1_0, conv_b1_0, conv_w1_1, conv_b1_1, conv_w1_2, conv_b1_2,
                     conv_w2_0, conv_b2_0, conv_w2_1, conv_b2_1, conv_w2_2, conv_b2_2,
                     conv_w3_0, conv_b3_0, conv_w3_1, conv_b3_1,
                     conv_w4_0, conv_b4_0, conv_w4_1, conv_b4_1,
                     conv_w5, conv_b5]

    def sample_generator(hidden_data):
        # linear stage (hidden_size => 2x2x512)
        seed = relu(T.dot(hidden_data, linear_w0) + linear_b0)
        seed = seed.reshape((seed.shape[0], num_gen_filters0, init_image_size, init_image_size))

        # deconv stage 0 (2x2x512=>4x4x512=>4x4x512=>4x4x512)
        h0_0 = relu(  deconv(seed, conv_w0_0, subsample=(2, 2), border_mode=(1, 1))+conv_b0_0.dimshuffle('x', 0, 'x', 'x'))
        h0_1 = relu(dnn_conv(h0_0, conv_w0_1, subsample=(1, 1), border_mode=(1, 1))+conv_b0_1.dimshuffle('x', 0, 'x', 'x'))
        h0_2 = relu(dnn_conv(h0_1, conv_w0_2, subsample=(1, 1), border_mode=(1, 1))+conv_b0_2.dimshuffle('x', 0, 'x', 'x'))

        # deconv stage 1 (4x4x512=>8x8x512=>8x8x512=>8x8x512)
        h1_0 = relu(  deconv(h0_2, conv_w1_0, subsample=(2, 2), border_mode=(1, 1))+conv_b1_0.dimshuffle('x', 0, 'x', 'x'))
        h1_1 = relu(dnn_conv(h1_0, conv_w1_1, subsample=(1, 1), border_mode=(1, 1))+conv_b1_1.dimshuffle('x', 0, 'x', 'x'))
        h1_2 = relu(dnn_conv(h1_1, conv_w1_2, subsample=(1, 1), border_mode=(1, 1))+conv_b1_2.dimshuffle('x', 0, 'x', 'x'))

        # deconv stage 2 (8x8x512=>16x16x256=>16x16x256=>16x16x256)
        h2_0 = relu(  deconv(h1_2, conv_w2_0, subsample=(2, 2), border_mode=(1, 1))+conv_b2_0.dimshuffle('x', 0, 'x', 'x'))
        h2_1 = relu(dnn_conv(h2_0, conv_w2_1, subsample=(1, 1), border_mode=(1, 1))+conv_b2_1.dimshuffle('x', 0, 'x', 'x'))
        h2_2 = relu(dnn_conv(h2_1, conv_w2_2, subsample=(1, 1), border_mode=(1, 1))+conv_b2_2.dimshuffle('x', 0, 'x', 'x'))

        # deconv stage 3 (16x16x256=>32x32x128=>32x32x128)
        h3_0 = relu(  deconv(h2_2, conv_w3_0, subsample=(2, 2), border_mode=(1, 1))+conv_b3_0.dimshuffle('x', 0, 'x', 'x'))
        h3_1 = relu(dnn_conv(h3_0, conv_w3_1, subsample=(1, 1), border_mode=(1, 1))+conv_b3_1.dimshuffle('x', 0, 'x', 'x'))

        # deconv stage 4 (32x32x128=>64x64x64=>64x64x64)
        h4_0 = relu(  deconv(h3_1, conv_w4_0, subsample=(2, 2), border_mode=(1, 1))+conv_b4_0.dimshuffle('x', 0, 'x', 'x'))
        h4_1 = relu(dnn_conv(h4_0, conv_w4_1, subsample=(1, 1), border_mode=(1, 1))+conv_b4_1.dimshuffle('x', 0, 'x', 'x'))

        # deconv output (64x64x64=>64x64x3)
        output = tanh(dnn_conv(h4_1, conv_w5, subsample=(1, 1), border_mode=(1, 1))+conv_b5.dimshuffle('x', 0, 'x', 'x'))

        return [T.flatten(h4_1, 2), T.flatten(h4_0, 2),
                T.flatten(h3_1, 2), T.flatten(h3_0, 2),
                T.flatten(h2_2, 2), T.flatten(h2_1, 2), T.flatten(h2_0, 2),
                T.flatten(h1_2, 2), T.flatten(h1_1, 2), T.flatten(h1_2, 2),
                T.flatten(h0_2, 2), T.flatten(h0_1, 2), T.flatten(h0_2, 2),
                T.flatten(seed, 2),
                output]

    return sample_generator, parameter_set


def set_updater_function(feature_extractor,
                         sample_generator,
                         generator_parameters,
                         generator_optimizer):
    # set input data, hidden data
    input_data  = T.tensor4(name='input_data',
                            dtype=theano.config.floatX)
    hidden_data = T.matrix(name='hidden_data',
                           dtype=theano.config.floatX)

    # extract feature from input data
    positive_features = feature_extractor(input_data)

    # sample data
    negative_features = sample_generator(hidden_data)
    negative_data     = negative_features[-1]
    negative_features = negative_features[:-1]

    # moment matching
    moment_match_cost = 0
    for i in xrange(len(positive_features)):
        pos_feat = positive_features[i]
        neg_feat = negative_features[i]
        moment_match_cost += T.sum(T.sqr(T.mean(pos_feat, axis=0)-T.mean(neg_feat, axis=0)))
        moment_match_cost += T.sum(T.sqr(T.mean(T.sqr(pos_feat), axis=0)-T.mean(T.sqr(neg_feat), axis=0)))

    pos_feat = T.flatten(input_data, 2)
    neg_feat = T.flatten(negative_data, 2)
    moment_match_cost += T.sum(T.sqr(T.mean(pos_feat, axis=0)-T.mean(neg_feat, axis=0)))
    moment_match_cost += T.sum(T.sqr(T.mean(T.sqr(pos_feat), axis=0)-T.mean(T.sqr(neg_feat), axis=0)))

    generator_updates = generator_optimizer(generator_parameters,
                                            moment_match_cost)

    # updater function input
    updater_function_inputs  = [input_data,
                               hidden_data]

    # updater function output
    updater_function_outputs = [moment_match_cost,
                                negative_data]

    # updater function
    updater_function = theano.function(inputs=updater_function_inputs,
                                       outputs=updater_function_outputs,
                                       updates=generator_updates,
                                       on_unused_input='ignore')
    return updater_function

def set_sampling_function(sample_generator):
    hidden_data = T.matrix(name='hidden_data',
                           dtype=theano.config.floatX)
    # sample data
    sample_data = sample_generator(hidden_data)[-1]

    # sampling function
    sampling_function = theano.function(inputs=[hidden_data],
                                        outputs=[sample_data,],
                                        on_unused_input='ignore')
    return sampling_function

def train_model(model_name,
                data_stream,
                num_hiddens,
                num_epochs,
                generator_optimizer):

    # set models
    print 'LOADING VGG'
    t=time()
    feature_extractor = load_vgg_feature_extractor()
    print '%.2f SEC '%(time()-t)
    sample_generator , generator_parameters = set_generator_model(num_hiddens)


    print 'COMPILING UPDATER AND SAMPLER'
    t=time()
    updater_function = set_updater_function(feature_extractor,
                                            sample_generator,
                                            generator_parameters,
                                            generator_optimizer)
    sampling_function = set_sampling_function(sample_generator)
    print '%.2f SEC '%(time()-t)

    # set fixed hidden data for sampling
    fixed_hidden_data  = floatX(np_rng.uniform(low=-1.0,
                                               high=1.0,
                                               size=(64*64, num_hiddens)))

    print 'START TRAINING'
    # for each epoch
    moment_cost_list = []
    batch_count = 0
    for e in xrange(num_epochs):
        # train phase
        batch_iters = data_stream.get_epoch_iterator()
        # for each batch
        for b, batch_data in enumerate(batch_iters):
            # set update function inputs
            input_data  = transform(batch_data[0])
            hidden_data = floatX(np_rng.uniform(low=-1.0, high=1.0, size=(input_data.shape[0], num_hiddens)))

            updater_inputs = [input_data,
                              hidden_data]
            updater_outputs = updater_function(*updater_inputs)
            moment_cost_list.append(updater_outputs[0])

            # batch count up
            batch_count += 1

            if batch_count%100==0:
                print '================================================================'
                print 'BATCH ITER #{}'.format(batch_count), model_name
                print '================================================================'
                print '   TRAIN RESULTS'
                print '================================================================'
                print '     moment matching cost     : ', moment_cost_list[-1]
                print '================================================================'

            if batch_count%1000==0:
                # sample data
                save_as = samples_dir + '/' + model_name + '_SAMPLES{}.png'.format(batch_count)
                sample_data = sampling_function(fixed_hidden_data)[0]
                sample_data = np.asarray(sample_data)
                color_grid_vis(inverse_transform(sample_data).transpose([0,2,3,1]), (16, 16), save_as)

                np.save(file=samples_dir + '/' + model_name +'_MOMENT_COST',
                        arr=np.asarray(moment_cost_list))

if __name__=="__main__":

    batch_size = 128
    num_epochs = 100
    _ , data_stream = faces(batch_size=batch_size)


    num_hiddens   = 100
    learning_rate = 1e-3
    l2_weight     = 1e-10

    generator_optimizer = Adagrad(lr=sharedX(learning_rate),
                                  regularizer=Regularizer(l2=l2_weight))

    model_test_name = model_name \
                      + '_HIDDEN{}'.format(int(num_hiddens)) \
                      + '_REG{}'.format(int(-np.log10(l2_weight))) \
                      + '_LR{}'.format(int(-np.log10(learning_rate))) \

    train_model(model_name=model_test_name,
                data_stream=data_stream,
                num_hiddens=num_hiddens,
                num_epochs=num_epochs,
                generator_optimizer=generator_optimizer)