"""
手写数字识别
"""

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow as tf
import os
import numpy as np
from PIL import Image
from common import cnn_layer, fnn_layer


def inference(input):
    input = tf.reshape(input, (-1, 28, 28, 1))
    with tf.variable_scope("cnn1", reuse=tf.AUTO_REUSE):
        cnn1 = cnn_layer(input, ksize=[5, 5, 1, 6], strides=[1, 1, 1, 1])
    with tf.variable_scope("cnn2", reuse=tf.AUTO_REUSE):
        cnn2 = cnn_layer(cnn1, ksize=[5, 5, 6, 16], strides=[1, 1, 1, 1])
    # 添加两个全连接层
    cnn2 = tf.reshape(cnn2, (-1, 4 * 4 * 16))
    with tf.variable_scope("fnn1", reuse=tf.AUTO_REUSE):
        fnn1 = fnn_layer(cnn2, 4 * 4 * 16, 120, activation=tf.nn.sigmoid)
    with tf.variable_scope("fnn2", reuse=tf.AUTO_REUSE):
        fnn2 = fnn_layer(fnn1, 120, 84, activation=tf.nn.sigmoid)
    # 添加一个输出层
    with tf.variable_scope("output_layer", reuse=tf.AUTO_REUSE):
        output_layer = fnn_layer(fnn2, 84, 10, activation=tf.nn.sigmoid)
    return output_layer
    # with tf.variable_scope("lay1", reuse=tf.AUTO_REUSE):
    #     weights = tf.get_variable(name="weights1",
    #                               shape=[784, 128],
    #                               initializer=tf.initializers.random_normal)
    #     biases = tf.get_variable(name="biases1",
    #                              shape=[128],
    #                              initializer=tf.initializers.constant)
    #     x = tf.matmul(input, weights) + biases
    #     x = tf.nn.sigmoid(x)
    # with tf.variable_scope("lay2", reuse=tf.AUTO_REUSE):
    #     weights2 = tf.get_variable(name="weights2",
    #                                shape=[128, 60],
    #                                initializer=tf.initializers.random_normal)
    #     biases2 = tf.get_variable(name="biases2",
    #                               shape=[60],
    #                               initializer=tf.initializers.constant)
    #     x2 = tf.matmul(x, weights2) + biases2
    #     x2 = tf.nn.sigmoid(x2)
    # with tf.variable_scope("lay3", reuse=tf.AUTO_REUSE):
    #     weights3 = tf.get_variable(name="weights3",
    #                                shape=[60, 10],
    #                                initializer=tf.initializers.random_normal)
    #     biases3 = tf.get_variable(name="biases3",
    #                               shape=[10],
    #                               initializer=tf.initializers.constant)
    #     x3 = tf.matmul(x2, weights3) + biases3
    #     x3 = tf.nn.sigmoid(x3)
    # return x3


def train(input_x, input_y, ephocs=10, batch_size=100):
    x = tf.placeholder("float", shape=[None, input_x.shape[1]], name="x-input")
    y = tf.placeholder("float", shape=[None, input_y.shape[1]], name="y-input")
    output = tf.nn.softmax(inference(x))
    # 定义损失函数
    global_steps = tf.Variable(0, trainable=False)
    cost = -tf.reduce_mean(y * tf.log(output))  # 逻辑回归的损失函数
    tf.summary.scalar("cost", cost)
    batches = input_x.shape[0] // batch_size
    if input_x.shape[0] % batch_size != 0:
        batches += 1
    learning_rate = tf.train.exponential_decay(0.03, global_steps, batches, 0.99, staircase=False)
    tf.summary.scalar("learning_rate", learning_rate)
    entropy_cost = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_steps)
    #  定义准确率的验证
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1)), "float"))
    tf.summary.scalar("accuracy", accuracy)
    merged_all = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter("logs", sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        for i in range(ephocs):
            for batch in range(batches):
                start = batch * batch_size % input_x.shape[0]
                end = min(start + batch_size, input_x.shape[0])
                _, merged = sess.run([entropy_cost, merged_all],
                                     feed_dict={x: input_x[start:end], y: input_y[start:end]})
            loss, predicated = sess.run([cost, accuracy], feed_dict={x: input_x, y: input_y})
            print(loss, predicated)
            train_writer.add_summary(merged, i)
            saver.save(sess, os.path.join("models", "mnist.ckpt"), global_step=global_steps)


def test_model(test_x, test_y, model_name):
    """
    测试模型
    :param test_x: 用来测试的样本特征
    :param test_y: 用来测试的样本标签
    :return: 模型的准确率
    """
    x = tf.placeholder("float", shape=[None, test_x.shape[1]], name="x-input")
    y = tf.placeholder("float", shape=[None, test_y.shape[1]], name="y-input")
    output = inference(x)
    output = tf.nn.softmax(output)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1)), "float"))
    with tf.Session()as sess:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join("models", model_name))
        return sess.run(accuracy, feed_dict={x: test_x, y: test_y})


def evaluate(input_x):
    x = tf.placeholder("float", shape=[None, input_x.shape[1]], name="x-input")
    output = inference(x)
    output = tf.argmax(tf.nn.softmax(output), axis=1)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join("models", "mnist.ckpt-550"))
        return sess.run(output, feed_dict={x: input_x})


mnist = read_data_sets("data\\", one_hot=True)
# train(mnist.train.images, mnist.train.labels)
with open(os.path.join("models", "checkpoint"), "r") as f:
    model_score = dict()
    models = [name.split(":")[1].lstrip() for name in f.readlines()[1:]]
    for model in models:
        model_name = "".join(list(model)[1:-2])
        score = test_model(mnist.test.images, mnist.test.labels, model_name)
        model_score[model_name] = score
    # 选择发布的模型
    max = 0
    release_model = ""
    for key in model_score.keys():
        if model_score[key] > max:
            max = model_score[key]
            release_model = key
    print(key)


def read_local_image(image_name):
    with open(image_name, "rb")as f:
        data = Image.open(f).convert("L")
        if data.width != 28 and data.height != 28:
            data = np.array(data.resize((28, 28)))
        data = 1.0 - np.array(data) / 255.0
        data = data.reshape(1, 784)
        print(evaluate(data))


read_local_image(os.path.join("data", "1.png"))
# read_local_image(os.path.join("data", "8.jpg"))
