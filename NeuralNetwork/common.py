# - *- coding:utf-8 -*-
import tensorflow as tf


def cnn_layer(input_tensor, ksize, strides, padding="VALID"):
    """
    往网络中添加一个卷积层
    :param input_tensor: 网络层的输入
    :param ksize: 卷积核的大小
    :param strides: 滑动的距离
    :param padding: 边的处理方式
    :return: output_tensor
    """
    weights = tf.get_variable("weights",
                              shape=ksize,
                              initializer=tf.initializers.random_normal)
    biases = tf.get_variable("biases",
                             shape=ksize[-1],
                             initializer=tf.initializers.constant)
    # 添加一个卷积
    output_tensor = tf.nn.conv2d(input_tensor, weights, strides, padding) + biases
    output_tensor = tf.nn.relu(output_tensor)
    # 添加一个池化
    output_tensor = tf.nn.max_pool(output_tensor, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
    return output_tensor


def fnn_layer(input_tensor, input_size, output_size, activation=None):
    """
    往网络中添加一个全连接层
    :param input_tensor: 网络层的输入
    :param input_size: 输入的大小
    :param output_size: 输出的大小
    :param activation: 当前网络层的激活函数
    :return: output_tensor
    """
    weights = tf.get_variable("weights",
                              shape=[input_size, output_size],
                              initializer=tf.initializers.random_normal)
    biases = tf.get_variable("biases",
                             shape=[output_size],
                             initializer=tf.initializers.constant)
    output_tensor = tf.matmul(input_tensor, weights) + biases
    if not activation:
        output_tensor = activation(output_tensor)
    return output_tensor
