'''
A Feed-forward Neural network with two layers for NER task
'''

from __future__ import print_function
import tensorflow as tf

class ModelHypPrms(object):
    '''Class for storing model hyperparameters'''
    def __init__(self, n_input, n_classes, hid_dim, lrn_rate):
        self.n_input = n_input
        self.n_classes = n_classes
        self.hid_dim = hid_dim
        self.lrn_rate = lrn_rate

class FFModel(object):
    '''Feed-Forward model'''
    def __init__(self, hps):
        # placeholders
        self.input_x = tf.placeholder(tf.float32, [None, hps.n_input])
        self.output_y = tf.placeholder(tf.float32, [None, hps.n_classes])
        self.dropout = tf.placeholder(tf.float32)
        # weights and biases
        weights_w1 = tf.Variable(tf.random_normal([hps.n_input, hps.hid_dim]))
        weights_w2 = tf.Variable(tf.random_normal([hps.hid_dim, hps.hid_dim]))
        weights_out = tf.Variable(tf.random_normal([hps.hid_dim, hps.n_classes]))
        biases_b1 = tf.Variable(tf.random_normal([hps.hid_dim])),
        biases_b2 = tf.Variable(tf.random_normal([hps.hid_dim])),
        biases_b3 = tf.Variable(tf.random_normal([hps.n_classes]))
        # operations for predictions
        layer_1 = tf.add(tf.matmul(self.input_x, weights_w1), biases_b1)
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, weights_w2), biases_b2)
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, self.dropout)
        self.pred = tf.matmul(layer_2, weights_out) + biases_b3
        # determine cost and optimize all variables
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                                   (logits=self.pred, labels=self.output_y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=hps.lrn_rate).minimize(self.cost)

