import tensorflow as tf
from tensorflow.contrib import slim
from collections import namedtuple


HParams = namedtuple('HParams',
                     'batch_size, min_lrn_rate, lrn_rate, '
                     'num_conv, weight_decay_rate, optimizer')


class IRCNN(object):

    def __init__(self, hps, image, label, mode):

        self.hps = hps
        self._image = image
        self.label = label
        self.mode = mode
        self.dilate = [1, 2, 3, 4, 3, 2, 1]

        if self.mode == 'Train':
            self.reuse = False
            self.is_training = True
        elif self.mode == 'Eval':
            self.reuse = False
            self.is_training = False


    def build_graph(self):

        self.global_step = tf.train.create_global_step()
        self._build_mode()

        if self.mode == 'Train':
            self._build_train_op()
        self.summaries = tf.summary.merge_all()


    def _build_mode(self):

        conv = slim.conv2d(self._image, 64, [3, 3], rate=self.dilate[0], activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(self.hps.weight_decay_rate), scope='conv_1', reuse=self.reuse)

        for i in range(2, self.hps.num_conv):

            conv = slim.conv2d(conv, 64, [3, 3], rate=self.dilate[i-1], activation_fn=None, weights_regularizer=slim.l2_regularizer(self.hps.weight_decay_rate), scope='conv_%d'%(i), reuse=self.reuse)
            conv = slim.batch_norm(conv, scale=True, activation_fn=tf.nn.relu, is_training=self.is_training)

        self.conv = slim.conv2d(conv, 3, [3, 3], rate=self.dilate[6], activation_fn=None, weights_regularizer=slim.l2_regularizer(self.hps.weight_decay_rate), scope='conv_%d'%(self.hps.num_conv), reuse=self.reuse)
        self.clear = self._image - self.conv

        if self.mode == 'Train':
            with tf.variable_scope('cost'):
                regu_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                content_cost = (1./self.hps.batch_size)*tf.nn.l2_loss(self.conv - self.label)
                self.cost = tf.add_n([content_cost] + regu_loss)
                tf.summary.scalar('cost', self.cost)


        with tf.variable_scope('input_image'):
            tf.summary.image('input_image', self._image)
        with tf.variable_scope('noisy'):
            tf.summary.image('noisy', self.conv)
        with tf.variable_scope('clear'):
            tf.summary.image('clear', self._image - self.conv)


    def _build_train_op(self):

        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.summary.scalar('learning_rate', self.lrn_rate)

        if self.hps.optimizer == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'mom':
            self.optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
        elif self.hps.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.lrn_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.cost, global_step=self.global_step)

        











