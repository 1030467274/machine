# - *- coding:utf-8 -*-
"""
手写数字识别
"""

from PIL import Image

import tensorflow as tf

from pandas.tests.extension.numpy_.test_numpy_nested import np

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import os


def inference(input):
    with tf.variable_scope("lay1", reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(name="weights1",

                                  shape=[784, 256],

                                  initializer=tf.initializers.random_normal)

        biases = tf.get_variable(name="biases1",

                                 shape=[256],

                                 initializer=tf.initializers.constant)

    x = tf.matmul(input, weights) + biases

    x = tf.nn.sigmoid(x)

    with tf.variable_scope("lay2", reuse=tf.AUTO_REUSE):
        weights2 = tf.get_variable(name="weights2",

                                   shape=[256, 10],

                                   initializer=tf.initializers.random_normal)

        biases2 = tf.get_variable(name="biases2",

                                  shape=[10],

                                  initializer=tf.initializers.constant)

    x2 = tf.matmul(x, weights2) + biases2

    return tf.nn.sigmoid(x2)


def train(input_x, input_y, ephocs=10, batch_size=100):
    x = tf.placeholder("float", shape=[None, input_x.shape[1]], name="x-input")

    y = tf.placeholder("float", shape=[None, input_y.shape[1]], name="y-input")

    output = tf.nn.softmax(inference(x))

    # 定义损失函数

    global_steps = tf.Variable(0, trainable=False)

    cost = -tf.reduce_mean(y * tf.log(output))

    tf.summary.scalar("cost", cost)

    batches = input_x.shape[0] // batch_size

    if input_x.shape[0] % batch_size != 0:
        batches += 1

    learning_rate = tf.train.exponential_decay(0.001, global_steps, batches, 0.99, staircase=False)

    tf.summary.scalar("learning_rate", learning_rate)

    entropy_cost = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_steps)

    # 定义准确率的验证

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

            loss, predicated = sess.run([cost, accuracy],

                                        feed_dict={x: input_x, y: input_y})

            print(loss, predicated)

            train_writer.add_summary(merged, i)

            saver.save(sess, os.path.join("models", "mnist.ckpt"), global_step=global_steps)


def test_model(test_x, test_y, model_name):
    """

    测试模型

    :param test_x:用来测试的样本特征

    :param test_y: 用来测试的样本标签

    :return: 模型的准确率

    """

    x = tf.placeholder("float", shape=[None, test_x.shape[1]], name="x-input")

    y = tf.placeholder("float", shape=[None, test_y.shape[1]], name="x-input")

    output = inference(x)

    output = tf.nn.softmax(output)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1)), "float"))

    with tf.Session() as sess:
        saver = tf.train.Saver()

        saver.restore(sess, os.path.join("models", model_name))

        return sess.run(accuracy, feed_dict={x: test_x, y: test_y})


def evaluate(input_x):
    x = tf.placeholder("float", shape=[None, input_x.shape[1]], name="x-input")

    output = inference(x)

    output = tf.argmax(tf.nn.softmax(output), axis=1)

    with tf.Session() as sess:
        saver = tf.train.Saver()

        saver.restore(sess, os.path.join("models", "mnist.ckpt-5500"))

        return sess.run(output, feed_dict={x: input_x})


def read_local_image(image_name):
    with open(image_name, "rb") as f:
        data = Image.open(f).convert("L")

        if data.width != 28 and data.height != 28:
            data = np.array(data.resize((28, 28)))

        data = 1.0 - np.array(data) / 255.0

        data = data.reshape(1, 784)

        print(evaluate(data))


read_local_image("1.png")

'''

mnist = read_data_sets("data\\", one_hot=True)

train(mnist.train.images, mnist.train.labels)

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

'''
