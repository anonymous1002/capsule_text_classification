from __future__ import division, print_function, unicode_literals
import argparse
import h5py
import numpy as np
import tensorflow as tf
from keras import utils
from keras import backend as K
from utils import _conv2d_wrapper
from layers import capsules_init, capsule_flatten, capsule_conv_layer, capsule_fc_layer
import tensorflow.contrib.slim as slim
from sklearn.utils import shuffle

tf.reset_default_graph()
np.random.seed(0)
tf.set_random_seed(0)

def baseline_model_kimcnn():
    pooled_outputs = []
    for i, filter_size in enumerate([3,4,5]):
        with tf.name_scope("conv-maxpool-%s" % filter_size):            
            filter_shape = [filter_size, 300, 1, 100]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[100]), name="b")
            conv = tf.nn.conv2d(X_embedding, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")            
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")            
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, args.max_sent - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)
    num_filters_total = 100 * 3
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    activations = tf.sigmoid(slim.fully_connected(h_pool_flat, args.num_classes, scope='final_layer', activation_fn=None))
    return activations
        
def capsule_model_v2():
    with tf.variable_scope('capsule_'+str(3)):   
        nets = _conv2d_wrapper(
            X_embedding, shape=[3, 300, 1, 32], strides=[1, 1, 1, 1], padding='VALID', 
            add_bias=True, activation_fn=tf.nn.relu, name='conv1'
        )
        tf.logging.info('output shape: {}'.format(nets.get_shape()))
        nets = capsules_init(nets, shape=[1, 1, 32, 32], strides=[1, 1, 1, 1], 
                             padding='VALID', pose_shape=16, add_bias=True, name='primary')                        
        nets = capsule_conv_layer(nets, shape=[3, 1, 32, 32], strides=[1, 1, 1, 1], iterations=3, name='conv3')
        nets = capsule_flatten(nets)
        poses, activations = capsule_fc_layer(nets, args.num_classes, 3, 'fc2') 
    return poses, activations

def capsule_model_v0():
    poses_list = []
    for _, ngram in enumerate([3,4,5]):
        with tf.variable_scope('capsule_'+str(ngram)):   
            nets = _conv2d_wrapper(
                X_embedding, shape=[ngram, 300, 1, 32], strides=[1, 1, 1, 1], padding='VALID', 
                add_bias=True, activation_fn=tf.nn.relu, name='conv1'
            )
            tf.logging.info('output shape: {}'.format(nets.get_shape()))
            nets = capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1], 
                                 padding='VALID', pose_shape=16, add_bias=True, name='primary')                      
            nets = capsule_conv_layer(nets, shape=[3, 1, 32, 32], strides=[1, 1, 1, 1], iterations=3, name='conv3')
            nets = capsule_flatten(nets)
            poses, _ = capsule_fc_layer(nets, args.num_classes, 3, 'fc2') 
            poses_list.append(poses)

    poses = tf.reduce_mean(tf.convert_to_tensor(poses_list), axis=0) 
    activations = K.sqrt(K.sum(K.square(poses), 2))    
    return poses, activations
