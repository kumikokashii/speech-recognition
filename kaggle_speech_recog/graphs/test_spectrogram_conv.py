from .useful_tf_graph import *
import tensorflow as tf
import math

class TestSpectrogramConv(UsefulTFGraph):
    def __init__(self, g_cnfg):
        super().__init__()
        self.build(g_cnfg)

    def weight_variable(self, shape, weights_stddev, name):
        initial = tf.truncated_normal(shape, stddev=weights_stddev)
        return tf.Variable(initial, name=name)
    
    def bias_variable(self, biases_initial, shape, name):
        initial = tf.constant(biases_initial, shape=shape)
        return tf.Variable(initial, name=name)

    def build(self, cnfg):
        self.cnfg = cnfg
        with self.as_default():
            global_step = tf.Variable(0, trainable=False)
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)

            self.X = tf.placeholder(tf.float32, [None, cnfg.X_img_h, cnfg.X_img_w])
            self.Y = tf.placeholder(tf.float32, [None, cnfg.Y_vector_len])
                    
            X_bn1 = tf.contrib.layers.batch_norm(inputs=tf.reshape(self.X, [-1, cnfg.X_img_h, cnfg.X_img_w, 1]),
                                                 updates_collections=None,
                                                 is_training=self.is_training,
                                                 scope='bn1')
            
            X_conv1 = tf.layers.conv2d(inputs=X_bn1,
                                       filters=cnfg.conv1_n_filters,
                                       kernel_size=cnfg.conv1_kernel_size,
                                       padding='same',
                                       activation=tf.nn.relu,
                                       name='conv1')
            
            X_pool1 = tf.layers.max_pooling2d(inputs=X_conv1,
                                              pool_size=cnfg.conv1_mp_size,
                                              strides=cnfg.conv1_mp_strides,
                                              padding='same',
                                              name='mp1')
            
            X_pool1_h = math.ceil(cnfg.X_img_h / cnfg.conv1_mp_strides)
            X_pool1_w = math.ceil(cnfg.X_img_w / cnfg.conv1_mp_strides)
            n_flat = X_pool1_h * X_pool1_w * cnfg.conv1_n_filters
            
            X1 = tf.reshape(X_pool1, [-1, n_flat])
            W1 = self.weight_variable([n_flat, cnfg.n_hidden], 0.015, 'W1')
            b1 = self.bias_variable(0.1, [cnfg.n_hidden], 'b1')
            XW1 = tf.matmul(X1, W1) + b1
            X2 = tf.nn.relu(XW1)

            W2 = self.weight_variable([cnfg.n_hidden, cnfg.Y_vector_len], 0.015, 'W2')
            b2 = self.bias_variable(0.1, [cnfg.Y_vector_len], 'b2')
            self.logits = tf.matmul(X2, W2) + b2
            
            # Accuracy
            correct_or_not = tf.cast(tf.equal(tf.argmax(self.Y, axis=1), tf.argmax(self.logits, axis=1)), tf.float32)
            # 1 if correct, 0 if not
            
            self.accuracy = tf.reduce_mean(correct_or_not)
            self.accuracy_batch_count = tf.reduce_sum(correct_or_not)

            # Logloss
            L = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.logits)
            self.logloss = tf.reduce_mean(L)
            self.logloss_batch_sum = tf.reduce_sum(L)
            
            # Optimization
            learning_rate = tf.train.exponential_decay(cnfg.lr_initial, global_step, 
                                                       cnfg.lr_decay_steps, cnfg.lr_decay_rate, 
                                                       staircase=True)

            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.logloss, global_step=global_step)

            # Tensorboard
            tf.summary.scalar('logloss', self.logloss)
            tf.summary.scalar('learning_rate', learning_rate)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

            self.summarizer = tf.summary.merge_all()
