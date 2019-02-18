import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.contrib.layers import flatten


class RecognizeTrafficSign(object):

    def __init__(self, image):
        image = cv2.resize(image, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
        image = [image]
        image = np.array(image)
        print(image.shape)
        self.image_gry = np.sum(image / 3, axis=3, keepdims=True)

        self.logits = lanet(self.image_gry)

        self.x = tf.placeholder(tf.float32, (None, 32, 32, 1))

    def lanet(x):
        mu = 0
        sigma = 0.1

        x = tf.cast(x, tf.float32)

        # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        w1 = tf.Variable(tf.random_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
        x = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID')

        b1 = tf.Variable(tf.zeros(6))

        x = tf.nn.bias_add(x, b1)

        # TODO: Activation.
        x = tf.nn.relu(x)

        # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # TODO: Layer 2: Convolutional. Output = 10x10x16.
        w2 = tf.Variable(tf.random_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
        x = tf.nn.conv2d(x, w2, strides=[1, 1, 1, 1], padding='VALID')
        b2 = tf.Variable(tf.zeros(16))
        x = tf.nn.bias_add(x, b2)

        # TODO: Activation.
        x = tf.nn.relu(x)

        # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # TODO: Flatten. Input = 5x5x16. Output = 400.
        x = flatten(x)

        # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
        w3 = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
        b3 = tf.Variable(tf.zeros(120))
        x = tf.add(tf.matmul(x, w3), b3)

        # TODO: Activation.
        x = tf.nn.relu(x)

        # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
        w4 = tf.Variable(tf.random_normal(shape=(120, 84), mean=mu, stddev=sigma))
        b4 = tf.Variable(tf.zeros(84))
        x = tf.add(tf.matmul(x, w4), b4)

        # TODO: Activation.
        x = tf.nn.relu(x)

        # TODO: Layer 5: Fully Connected. Input = 84. Output = 62.

        w5 = tf.Variable(tf.random_normal(shape=(84, 62), mean=mu, stddev=sigma))
        b5 = tf.Variable(tf.zeros(62))
        logits = tf.add(tf.matmul(x, w5), b5)

        return logits


    def predict(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver3 = tf.train.import_meta_graph('./lanet.meta')
            saver3.restore(sess, "./lanet")
            prediction = evaluate()
            print("prediction: ".format(prediction))

        return prediction

    def evaluate(self):
        sess = tf.get_default_session()
        pred = sess.run(self.logits, feed_dict={self.x: self.image_gry})
        print('PRED : ', np.argmax(pred))

        return pred


# ==================================================

"""

def predict(image):

    image = cv2.resize(image, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
    image = [image]
    image = np.array(image)
    print(image.shape)
    image_gry = np.sum(image/3, axis=3, keepdims=True)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver3 = tf.train.import_meta_graph('./lanet.meta')
        saver3.restore(sess, "./lanet")
        prediction = evaluate(image_gry)
        print("prediction: ".format(prediction))

    return prediction

def lanet(x) :
    mu = 0
    sigma = 0.1

    x = tf.cast(x, tf.float32)

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    w1 = tf.Variable(tf.random_normal(shape=(5,5,1,6),mean=mu, stddev=sigma))
    x = tf.nn.conv2d(x,w1,strides=[1,1,1,1], padding='VALID')

    b1 = tf.Variable(tf.zeros(6))

    x = tf.nn.bias_add(x,b1)

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    w2 = tf.Variable(tf.random_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    x = tf.nn.conv2d(x, w2, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16))
    x = tf.nn.bias_add(x, b2)

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    x = flatten(x)

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    w3 = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    b3 = tf.Variable(tf.zeros(120))
    x = tf.add(tf.matmul(x, w3), b3)

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    w4 = tf.Variable(tf.random_normal(shape=(120, 84), mean=mu, stddev=sigma))
    b4 = tf.Variable(tf.zeros(84))
    x = tf.add(tf.matmul(x, w4), b4)

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 62.


    w5 = tf.Variable(tf.random_normal(shape=(84, 62), mean=mu, stddev=sigma))
    b5 = tf.Variable(tf.zeros(62))
    logits = tf.add(tf.matmul(x, w5), b5)

    return logits



def evaluate(X_data):
    sess = tf.get_default_session()

    x = tf.placeholder(tf.float32, (None, 32, 32, 1))

    pred = sess.run(logits, feed_dict={x: X_data})
    print('PRED : ', np.argmax(pred))

    return pred

"""
