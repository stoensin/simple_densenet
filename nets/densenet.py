"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net
def transition_layer(net,in_size, scope):
    """
    The transition layers used in our experiments consist of a batch normalization layer 
    and an 1×1 convolutional layer 
    followed by a 2×2 average pooling layer.
    """
    net= slim.batch_norm(net, scope=scope + '_bn')
    net= slim.conv2d(net, in_size, [1,1], scope=scope + '_conv1x1')
    print(net.get_shape())
    net= slim.avg_pool2d(net, [2, 2], stride=2, scope=scope+'_pool_2x2')
    
    return net

def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.get_shape()[1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,keep_prob=dropout_keep_prob)) as ssc:
            # pass
            #images: A batch of `Tensors` of size [batch_size, height, width, channels].
            ##########################
            
            # code start
            
            #  conv1: The initial convolution layer comprises 2k convolutions of size 7×7 with stride 2
            net= slim.conv2d(images, growth*2, [7,7],stride=2, scope=scope + '_conv7x7')
            end_points[scope + '_conv7x7'] = net
            print(net.get_shape())
            #  pool1
            net= slim.max_pool2d(net, [3,3],stride=2,padding='SAME', scope=scope + '_pool3x3')
            end_points[scope + '_pool3x3'] = net
            print(net.get_shape())
            # denseblock1
            net= block(net, 6, growth, scope= 'denseblock1')
            end_points['denseblock1'] = net
            print(net.get_shape())
            # transition layer1
            net= transition_layer(net, reduce_dim(net), scope='Transition_Layer1')
            end_points['Transition_Layer1'] = net
            print(net.get_shape())
            # denseblock2
            net= block(net, 12, growth, scope= 'denseblock2')
            end_points['denseblock2'] = net
            print(net.get_shape())
            # transition layer2
            net= transition_layer(net, reduce_dim(net), scope='Transition_Layer2')
            end_points['Transition_Layer2'] = net
            print(net.get_shape())
            # denseblock3
            net= block(net, 24, growth, scope= 'denseblock3')
            end_points['denseblock3'] = net
            print(net.get_shape())
            # transition layer3
            net= transition_layer(net, reduce_dim(net), scope='Transition_Layer3')
            end_points['Transition_Layer3'] = net
            print(net.get_shape())
            # denseblock4
            net= block(net, 16, growth, scope= 'denseblock4')
            end_points['denseblock4'] = net
            print(net.get_shape())
            
            # Global average pool
            net= slim.avg_pool2d(net, 1, [7,7], scope=scope + '_pool7x7')
            net= slim.flatten(net)
            print(net.get_shape())
            logits = slim.fully_connected(net, num_classes, scope='output',activation_fn=tf.nn.softmax)
            print(logits.get_shape())
            end_points['logits'] = logits
            
            # code end
            ##########################

    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
