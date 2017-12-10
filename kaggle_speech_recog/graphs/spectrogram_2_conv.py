from .useful_tf_graph import *
import tensorflow as tf

class Spectrogram2Conv(UsefulTFGraph):
    def __init__(self, g_cnfg):
        super().__init__()
        self.build(g_cnfg)

    def get_weight_tensor(self, shape, stddev):
        return tf.get_variable(name='W', shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    
    def get_bias_tensor(self, shape, value):
        return tf.get_variable(name='b', shape=shape, initializer=tf.constant_initializer(value=value))

    def apply_batch_normalize(self, inputs, is_training):
        return tf.contrib.layers.batch_norm(inputs=inputs,
                                            updates_collections=None,
                                            is_training=is_training,
                                            scope='bn')

    def apply_convolution(self, inputs, n_filters, kernel_size):
        return tf.layers.conv2d(inputs=inputs,
                                filters=n_filters,
                                kernel_size=kernel_size,
                                padding='same',
                                activation=tf.nn.relu,
                                name='conv')
        
    def apply_max_pooling(self, inputs, mp_size, strides):
        return tf.layers.max_pooling2d(inputs=inputs,
                                       pool_size=mp_size,
                                       strides=strides,
                                       padding='same',
                                       name='mp')
    
    def apply_bn_conv_mp(self, inputs, is_training, conv_n_filters, conv_kernel_size, mp_size, mp_strides):
        bn_ = self.apply_batch_normalize(inputs, is_training)
        conv_ = self.apply_convolution(bn_, conv_n_filters, conv_kernel_size)
        mp_ = self.apply_max_pooling(conv_, mp_size, mp_strides)
        return mp_
    
    def apply_bn_dr_XWplusb(self, X, is_training, dr_keep_prob, W_shape, W_stddev, b_shape, b_value, skip_relu=False):
        bn_ = self.apply_batch_normalize(X, is_training)
        dropout_ = tf.nn.dropout(bn_, keep_prob=dr_keep_prob)
        W = self.get_weight_tensor(W_shape, W_stddev)
        b = self.get_bias_tensor(b_shape, b_value)
        XWplusb = tf.matmul(dropout_, W) + b
        
        if skip_relu:
            return XWplusb
        else:
            return tf.nn.relu(XWplusb)

    def build(self, cnfg):
        self.cnfg = cnfg
        with self.as_default():
            global_step = tf.Variable(0, trainable=False)
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)

            self.X = tf.placeholder(tf.float32, [None, cnfg.X_img_h, cnfg.X_img_w])
            self.Y = tf.placeholder(tf.float32, [None, cnfg.Y_vector_len])
            
            with tf.variable_scope('conv1'):
                X_conv1 = self.apply_bn_conv_mp(tf.reshape(self.X, [-1, cnfg.X_img_h, cnfg.X_img_w, 1]),
                                                is_training=self.is_training,
                                                conv_n_filters=cnfg.conv1_n_filters, 
                                                conv_kernel_size=cnfg.conv1_kernel_size, 
                                                mp_size=cnfg.conv1_mp_size, 
                                                mp_strides=cnfg.conv1_mp_strides)
                
            with tf.variable_scope('conv2'):
                X_conv2 = self.apply_bn_conv_mp(X_conv1,
                                                is_training=self.is_training,
                                                conv_n_filters=cnfg.conv2_n_filters, 
                                                conv_kernel_size=cnfg.conv2_kernel_size, 
                                                mp_size=cnfg.conv2_mp_size, 
                                                mp_strides=cnfg.conv2_mp_strides)                

            X1 = tf.reshape(X_conv2, [-1, cnfg.n_flat])
            
            with tf.variable_scope('flat1'):
                X2 = self.apply_bn_dr_XWplusb(X=X1, is_training=self.is_training, dr_keep_prob=self.keep_prob,
                                              W_shape=[cnfg.n_flat, cnfg.n_hidden1], W_stddev=0.015,
                                              b_shape=[cnfg.n_hidden1], b_value=0.1)
                
            with tf.variable_scope('flat2'):
                X3 = self.apply_bn_dr_XWplusb(X=X2, is_training=self.is_training, dr_keep_prob=self.keep_prob,
                                              W_shape=[cnfg.n_hidden1, cnfg.n_hidden2], W_stddev=0.015,
                                              b_shape=[cnfg.n_hidden2], b_value=0.1)
                
            with tf.variable_scope('flat3'):
                self.logits = self.apply_bn_dr_XWplusb(X=X3, is_training=self.is_training, dr_keep_prob=self.keep_prob,
                                                       W_shape=[cnfg.n_hidden2, cnfg.Y_vector_len], W_stddev=0.015,
                                                       b_shape=[cnfg.Y_vector_len], b_value=0.1,
                                                       skip_relu=True)
            
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
