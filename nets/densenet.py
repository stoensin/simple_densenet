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
    net= slim.conv2d(net, in_size, [1,1], scope=scope + '_conv1x1')
    net= slim.batch_norm(net, scope=scope + '_bn')
    net= tf.nn.relu(net)
    net = slim.dropout(net, keep_prob=0.5,scope=scope + '_dropout')
    net= slim.avg_pool2d(net, [2, 2], stride=2, scope=scope+'_pool_2x2')

    return net

def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.2,
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
    growth = 40
    compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.get_shape()[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,keep_prob=dropout_keep_prob)) as ssc:
            # pass
            #images: A batch of `Tensors` of size [batch_size, height, width, channels].
            ##########################
            # network architecture work on cifar10 dataset:
            # (batch, 112, 112, 48)  conv7x7
            # (batch, 56, 56, 48)    max_pool3x3
            # (batch, 56, 56, 192)   denseblock1
            # (batch, 56, 56, 96)    transition conv1x1
            # (batch, 28, 28, 96)    transition avg_pool2x2
            # (batch, 28, 28, 384)   denseblock2
            # (batch, 28, 28, 192)   transition conv1x1
            # (batch, 14, 14, 192)   transition avg_pool2x2
            # (batch, 14, 14, 768)   denseblock3
            # (batch, 14, 14, 384)   transition conv1x1
            # (batch, 7, 7, 384)     transition avg_pool2x2
            # (batch, 7, 7, 768)     denseblock4
            # (batch, 768)           global average pool7x7
            # (batch, 10)            classifier-layer

            # 从网络结构来看,网络变得越来越窄,而这样越来越窄的效果就是 densenet 结构设计的精妙之处.

            # code start

            #  The initial convolution layer comprises 2k(k:growth) convolutions of size 7×7 with stride 2
            #  Before entering the first dense block, a convolution with 16 (or twice the growth rate for DenseNet-BC) output channels is performed on the input images
            net= slim.conv2d(images, growth*2, [7,7],stride=2, scope=scope + '_conv7x7')
            end_points[scope + '_conv7x7'] = net

            net= slim.max_pool2d(net, [3,3],stride=2,padding='SAME', scope=scope + '_pool3x3')
            end_points[scope + '_max_pool3x3'] = net

           # denseblock1:
           # denseblock的输出H*W维度不会变化,channel会根据growth*num增长
           # 这里的 growth 就起到一个限制网络变宽,对参数保持一个限制,而又在深度上保持按比例的增长,其内部的bottleneck实现又对channel数量进行了降低,在网络结构中处处都对参数进行着控制,
           # 而反应到特征学习上来说,就是 each layer 只学习少量特征,这就减少了参数量,降低了网络的对于计算的需求;
           # 对于网络的目标而言,denseblock内的复合结构对于输入的特征学习的方式-特征复用,每一层网络的输出都会被用于后面各层网络的计算;
           # 而在该结构内各层连接的数量有 (l(l+1))//2 个,如果以l=50计算,连接数就有1275了,所以就有密集连接的说法;
            net= block(net, 6, growth, scope= scope + 'denseblock1')
            end_points[scope +'denseblock1'] = net

            # transition layer1
            # : To further improve model compactness, we can reduce the number of feature-maps at transition layers
            #   经过 denseblock对网络输出维度 channel 的提升后,transition 则起到降维(对 Feature Map 数量和尺寸)作用,减少网络的计算量
            net= transition_layer(net, reduce_dim(net), scope=scope + 'Transition_Layer1')
            end_points[scope +'Transition_Layer1'] = net

            # denseblock2
            net= block(net, 12, growth, scope= scope + 'denseblock2')
            end_points[scope +'denseblock2'] = net

            # transition layer2
            net= transition_layer(net, reduce_dim(net), scope=scope + 'Transition_Layer2')
            end_points[scope +'Transition_Layer2'] = net

            # denseblock3
            net= block(net, 24, growth, scope=scope + 'denseblock3')
            end_points[scope +'denseblock3'] = net

            # transition layer3
            net= transition_layer(net, reduce_dim(net), scope=scope + 'Transition_Layer3')
            end_points[scope +'Transition_Layer3'] = net

            # denseblock4
            net= block(net, 16, growth, scope=scope + 'denseblock4')
            end_points[scope +'denseblock4'] = net

            net= slim.batch_norm(net, scope=scope + '_bn')
            net= tf.nn.relu(net)
            # Global average pool:对网络传输过来的特征图各层都做avg_pool,且pool的siez就和特征图同大小
            # 原理和用FC全连接层类似,但这个方式带来的参数量很小,降低计算量,还相当于对特征直接进行了粗粒度分类,
            net= slim.avg_pool2d(net, int(net.get_shape()[1]), stride=1, scope=scope + '_gap_pool7x7')
            net= slim.flatten(net)
            end_points[scope + '_gap_pool7x7'] = net

            # softmax classifier
            logits = slim.fully_connected(net, num_classes, scope=scope + 'output', activation_fn=tf.nn.softmax)
            end_points[scope + 'logits'] = logits

            # code end
            ##########################

    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.5):
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
