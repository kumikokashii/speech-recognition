from .useful_tf_graph import *
import tensorflow as tf

class SpectrogramConcatConvLSTM(UsefulTFGraph):
    def __init__(self, g_cnfg):
        super().__init__()
        self.build(g_cnfg)

    def get_weight_tensor(self, shape, stddev=0.015):
        return tf.get_variable(name='W', shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    
    def get_bias_tensor(self, shape, value=0.1):
        return tf.get_variable(name='b', shape=shape, initializer=tf.constant_initializer(value=value))

    def apply_batch_normalize(self, inputs):
        return tf.contrib.layers.batch_norm(inputs=inputs,
                                            updates_collections=None,
                                            is_training=self.is_training,
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
    
    def apply_bn_conv_mp(self, inputs, conv_n_filters, conv_kernel_size, mp_size, mp_strides):
        bn_ = self.apply_batch_normalize(inputs)
        conv_ = self.apply_convolution(bn_, conv_n_filters, conv_kernel_size)
        mp_ = self.apply_max_pooling(conv_, mp_size, mp_strides)
        return mp_
    
    def apply_bn_dr_XWplusb(self, X, W_shape, W_stddev=0.015, b_value=0.1, skip_relu=False):
        bn_ = self.apply_batch_normalize(X)
        dropout_ = tf.nn.dropout(bn_, self.keep_prob)
        W = self.get_weight_tensor(W_shape, W_stddev)
        b = self.get_bias_tensor([W_shape[1]], b_value)
        XWplusb = tf.matmul(dropout_, W) + b
        
        if skip_relu:
            return XWplusb
        else:
            return tf.nn.relu(XWplusb)

    def apply_lstm(self, inputs, input_dim_per_time, state_sizes, dr_dict):
        batch_size = tf.shape(inputs)[0]
        input_sizes = [input_dim_per_time] + state_sizes

        cell_list = []
        initial_state_list = []                
        for i in range(len(state_sizes)):
            cell = tf.contrib.rnn.BasicLSTMCell(state_sizes[i])  # forget_bias=1.0
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_size=input_sizes[i], dtype=tf.float32,
                                                 input_keep_prob=(self.keep_prob if dr_dict['in']=='y' else 1.0),
                                                 output_keep_prob=(self.keep_prob if dr_dict['out']=='y' else 1.0),
                                                 state_keep_prob=(self.keep_prob if dr_dict['state']=='y' else 1.0),
                                                 variational_recurrent=True)                    
            initial_state = cell.zero_state(batch_size, tf.float32)

            cell_list.append(cell)
            initial_state_list.append(initial_state)

        multi_cell = tf.contrib.rnn.MultiRNNCell(cells=cell_list)
        outputs, final_state = tf.nn.dynamic_rnn(multi_cell, inputs, initial_state=tuple(initial_state_list))
        # outputs is (batch size) x (length of time) x (state size)

        final_output = outputs[:, -1, :]
        return final_output

    def apply_bn_lstm(self, inputs, lstm_args):
        # inputs' shape is (batch size) x (length of time) x (dim of data at each time)
        bn_ = self.apply_batch_normalize(inputs)
        lstm_ = self.apply_lstm(bn_, *lstm_args)
        return lstm_    
    
    def prebuild(self, cnfg):
        cnfg.conv1_args = (cnfg.conv1_n_filters, cnfg.conv1_kernel_size, cnfg.conv1_mp_size, cnfg.conv1_mp_strides)
        cnfg.conv2_args = (cnfg.conv2_n_filters, cnfg.conv2_kernel_size, cnfg.conv2_mp_size, cnfg.conv2_mp_strides)
        cnfg.lstm_args = (cnfg.X_img_h, cnfg.lstm_state_sizes, cnfg.lstm_dropout)
        return cnfg
        
    def build(self, cnfg):
        cnfg = self.prebuild(cnfg)
        self.cnfg = cnfg
        
        with self.as_default():
            global_step = tf.Variable(0, trainable=False)
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)

            self.X = tf.placeholder(tf.float32, [None, cnfg.X_img_h, cnfg.X_img_w])
            self.Y = tf.placeholder(tf.float32, [None, cnfg.Y_vector_len])
            
            with tf.variable_scope('conv1'):
                X_conv1 = self.apply_bn_conv_mp(tf.reshape(self.X, [-1, cnfg.X_img_h, cnfg.X_img_w, 1]), *cnfg.conv1_args)
                
            with tf.variable_scope('conv2'):
                X_conv2 = self.apply_bn_conv_mp(X_conv1, *cnfg.conv2_args)                
            
            with tf.variable_scope('lstm'):                
                X_lstm = self.apply_bn_lstm(tf.transpose(self.X, [0, 2, 1]), cnfg.lstm_args)
           
            X_conv_flat = tf.reshape(X_conv2, [-1, cnfg.n_conv_flat])
            X_lstm_flat = tf.reshape(X_lstm, [-1, cnfg.lstm_state_sizes[-1]])
            X1 = tf.concat([X_conv_flat, X_lstm_flat], axis=1)
            
            with tf.variable_scope('flat1'):
                X2 = self.apply_bn_dr_XWplusb(X1, W_shape=[cnfg.n_flat, cnfg.n_hidden1])
                
            with tf.variable_scope('flat2'):
                X3 = self.apply_bn_dr_XWplusb(X2, W_shape=[cnfg.n_hidden1, cnfg.n_hidden2])
                
            with tf.variable_scope('flat3'):
                self.logits = self.apply_bn_dr_XWplusb(X3, W_shape=[cnfg.n_hidden2, cnfg.Y_vector_len],
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
