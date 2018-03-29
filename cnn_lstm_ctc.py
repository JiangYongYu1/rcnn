import tensorflow as tf
import utils
from tensorflow.python.training import moving_averages
import numpy as np

FLAGS = utils.FLAGS
num_classes = utils.num_classes
import pandas as pd


class LSTMOCR(object):
    def __init__(self, mode):
        self.mode = mode
        # image
        self.inputs = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
        # SparseTensor required by ctc_loss op
        self.labels = tf.sparse_placeholder(tf.int32)
        # 1d array of size [batch_size]
        # l2
        self._extra_train_ops = []

    def build_graph(self):
        self._build_model()
        self._build_train_op()

        self.merged_summay = tf.summary.merge_all()

    def _build_model(self):
        filters = [64, 128, 256, 512]
        strides = [1, 2]

        with tf.variable_scope('cnn'):
            with tf.variable_scope('unit-1'):
                x = self._conv2d(self.inputs, 'cnn-1', 3, 1, filters[0], strides[0])
                x = self._batch_norm('bn1', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides[1])

            with tf.variable_scope('unit-2'):
                x = self._conv2d(x, 'cnn-2', 3, filters[0], filters[1], strides[0])
                x = self._batch_norm('bn2', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides[1])

            with tf.variable_scope('unit-3'):
                x = self._conv2d(x, 'cnn-3', 3, filters[1], filters[2], strides[0])
                x = self._batch_norm('bn3', x)
                x = self._leaky_relu(x, 0.01)

            with tf.variable_scope('unit-4'):
                x = self._conv2d(x, 'cnn-4', 3, filters[2], filters[2], strides[0])
                x = self._batch_norm('bn4', x)
                x = self._leaky_relu(x, 0.01)
                x = tf.layers.max_pooling2d(inputs=x, pool_size=[1, 2], strides=2)

            with tf.variable_scope('unit-5'):
                x = self._conv2d(x, 'cnn-5', 3, filters[2], filters[3], strides[0])
                x = self._batch_norm('bn5', x)
                x = self._leaky_relu(x, 0.01)

            with tf.variable_scope('unit-6'):
                x = self._conv2d(x, 'cnn-6', 2, filters[3], filters[3], strides[0])
                x = self._batch_norm('bn6', x)
                x = self._leaky_relu(x, 0.01)
                x = tf.layers.max_pooling2d(inputs=x, pool_size=[1, 2], strides=2)

            with tf.variable_scope('unit-8'):
                x = self._conv2d(x, 'cnn-8', 2, filters[3], filters[3], strides[0])
                x = self._batch_norm('bn8', x)
                x = self._leaky_relu(x, 0.01)


        with tf.variable_scope('lstm'):
            # [batch_size, max_stepsize, num_feature]
            print(x.get_shape().as_list())
            batch_size, height, width, channels = x.get_shape().as_list()
            self.seq_len = tf.fill([tf.shape(x)[0]], width)
            # x = tf.transpose(x, [0, 2, 1])  # batch_size * 64 * 48
            x = tf.reshape(x, [-1, width, height*channels])
            print(x.get_shape())
            # x = tf.transpose(x, [1, 0, 2])
            # x = tf.unstack(x)
            for l in range(2):
                fw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)
                bw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)
                (x, _)= tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, self.seq_len,
                                                                  dtype=tf.float32, scope=str(l))
                x = tf.concat(x, 2)

            W = tf.get_variable(name='W',
                                shape=[2*FLAGS.num_hidden, num_classes],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='b',
                                shape=[num_classes],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())
            x = tf.reshape(x, [-1, 2*FLAGS.num_hidden])
            self.logits = tf.matmul(x, W) + b
            # Reshaping back to the original shape
            self.logits = tf.reshape(self.logits, [batch_size, -1, num_classes])
            # Time major
            self.logits = tf.transpose(self.logits, (1, 0, 2))


    def _build_train_op(self):
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.global_step = tf.Variable(0, trainable=False)

        self.loss = tf.nn.ctc_loss(labels=self.labels,
                                   inputs=self.logits,
                                   sequence_length=self.seq_len)
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('cost', self.cost)

        self.lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   self.global_step,
                                                   FLAGS.decay_steps,
                                                   FLAGS.decay_rate,
                                                   staircase=True)

        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lrn_rate,
        #                                            momentum=FLAGS.momentum).minimize(self.cost,
        #                                                                              global_step=self.global_step)
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lrn_rate,
        #                                             momentum=FLAGS.momentum,
        #                                             use_nesterov=True).minimize(self.cost,
        #                                                                         global_step=self.global_step)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
                                                beta1=FLAGS.beta1,
                                                beta2=FLAGS.beta2).minimize(self.loss,
                                                                            global_step=self.global_step)
        train_ops = [self.optimizer] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len,merge_repeated=False)
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits,
                                                                    self.seq_len,
                                                                    merge_repeated=False)
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)

    def _conv2d(self, x, name, filter_size, in_channels, out_channels, strides):
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='DW',
                                     shape=[filter_size, filter_size, in_channels, out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())

            b = tf.get_variable(name='bais',
                                shape=[out_channels],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')

        return tf.nn.bias_add(con2d_op, b)

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)
            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            x_bn = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            x_bn.set_shape(x.get_shape())

            return x_bn

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _max_pool(self, x, ksize, strides):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, strides, strides, 1],
                              padding='SAME',
                              name='max_pool')
