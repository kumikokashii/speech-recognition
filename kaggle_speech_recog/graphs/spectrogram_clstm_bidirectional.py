from .useful_tf_graph import *
import tensorflow as tf

class SpectrogramCLSTMBidirectional(UsefulTFGraph):
    def __init__(self, g_cnfg):
        super().__init__()
        self.build(g_cnfg)

    def random_modify(self, image, b_max_delta, c_lower, c_upper):
        image = tf.image.random_brightness(image, max_delta=b_max_delta)
        image = tf.image.random_contrast(image, lower=c_lower, upper=c_upper)
        return image      

    def batch_random_modify(self, inputs, img_h, img_w, b_max_delta, c_lower, c_upper):
        temp = tf.reshape(inputs, [-1, img_h, img_w, 1])
        temp = tf.map_fn(lambda x: self.random_modify(x, b_max_delta, c_lower, c_upper), temp)
        return tf.reshape(temp, [-1, img_h, img_w])    
    
    def get_weight_tensor(self, shape, stddev):
        return tf.get_variable(name='W', shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    
    def get_bias_tensor(self, shape, value):
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

    def reshape_for_lstm(self, inputs, n_time, n_per_time):
        # From conv output to lstm input
        temp = tf.transpose(inputs, [0, 2, 1, 3])  # Flip height and width  
        return tf.reshape(temp, [-1, n_time, n_per_time]) # Collapse channels
                
    def get_lstm_outputs(self, inputs, dim_per_time, state_sizes):
        # inputs.shape = (batch size) x (length of time) x (dim of data at each time)
        batch_size = tf.shape(inputs)[0]
        
        bn_ = self.apply_batch_normalize(inputs)

        cell_list = []
        initial_state_list = []
        input_sizes = [dim_per_time] + state_sizes
        for i in range(len(state_sizes)):
            cell = tf.contrib.rnn.BasicLSTMCell(state_sizes[i])  # forget_bias=1.0
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_size=input_sizes[i], dtype=tf.float32,
                                                 input_keep_prob=self.keep_prob,
                                                 output_keep_prob=1.0,
                                                 state_keep_prob=1.0,
                                                 variational_recurrent=True)
            cell_list.append(cell)

            initial_state = cell.zero_state(batch_size, tf.float32)                    
            initial_state_list.append(initial_state)

        multi_cell = tf.contrib.rnn.MultiRNNCell(cells=cell_list)
        outputs, final_state = tf.nn.dynamic_rnn(multi_cell, bn_, initial_state=tuple(initial_state_list))        
        # outputs is (batch size) x (length of time) x (lstm state size)
        
        return outputs        

    def get_bidirectional_lstm_outputs(self, inputs, dim_per_time, state_sizes):
        # inputs.shape = (batch size) x (length of time) x (dim of data at each time)
        batch_size = tf.shape(inputs)[0]
        
        bn_ = self.apply_batch_normalize(inputs)

        cell_list = []
        initial_state_list = []
        input_sizes = [dim_per_time] + state_sizes
        for i in range(len(state_sizes)):
            cell = tf.contrib.rnn.BasicLSTMCell(state_sizes[i])  # forget_bias=1.0
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_size=input_sizes[i], dtype=tf.float32,
                                                 input_keep_prob=self.keep_prob,
                                                 output_keep_prob=1.0,
                                                 state_keep_prob=1.0,
                                                 variational_recurrent=True)
            cell_list.append(cell)

            initial_state = cell.zero_state(batch_size, tf.float32)                    
            initial_state_list.append(initial_state)

        multi_cell = tf.contrib.rnn.MultiRNNCell(cells=cell_list)
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=multi_cell, cell_bw=multi_cell, 
                                                                 initial_state_fw=tuple(initial_state_list), 
                                                                 initial_state_bw=tuple(initial_state_list),
                                                                 inputs=bn_)
        # outputs is (output_fw, output_bw). Each of shape (batch size) x (length of time) x (lstm state size)
        return tf.concat(outputs, 2)
    
    def flatten_last_output(self, lstm_outputs, n_flat):                        
        return tf.reshape(lstm_outputs[:, -1, :], [-1, n_flat])
    
    def flatten_maxpool_output(self, lstm_outputs, n, n_flat):
        last_n_outputs = lstm_outputs[:, -n:, :]
        maxpool_output = tf.map_fn(lambda x: tf.reduce_max(x, axis=0), last_n_outputs)
        return tf.reshape(maxpool_output, [-1, n_flat])
    
    def apply_bn_dr_XWplusb(self, X, W_shape, W_stddev=0.015, b_value=0.1, skip_relu=False):
        bn_ = self.apply_batch_normalize(X)
        dropout_ = tf.nn.dropout(bn_, keep_prob=self.keep_prob)
        W = self.get_weight_tensor(W_shape, W_stddev)
        b = self.get_bias_tensor(W_shape[1], b_value)
        XWplusb = tf.matmul(dropout_, W) + b
        
        if skip_relu:
            return XWplusb
        else:
            return tf.nn.relu(XWplusb)

    def prebuild(self, cnfg):
        cnfg.random_modify_args = (cnfg.X_img_h, cnfg.X_img_w,
                                   cnfg.random_modify_brightness_max_delta, 
                                   cnfg.random_modify_contrast_lower, cnfg.random_modify_contrast_upper)
        cnfg.n_flat = 2 * cnfg.dim_time_after_conv * cnfg.lstm_state_sizes[-1]
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
            
            with tf.variable_scope('random_modify'):
                self.X = tf.cond(self.is_training,
                                 lambda: self.batch_random_modify(self.X, *cnfg.random_modify_args), 
                                 lambda: tf.identity(self.X))
            
            with tf.variable_scope('conv'):
                conv_output = self.apply_bn_conv_mp(tf.reshape(self.X, [-1, cnfg.X_img_h, cnfg.X_img_w, 1]), 
                                                    cnfg.conv_n_filters, cnfg.conv_kernel_size, 
                                                    cnfg.conv_mp_size, cnfg.conv_mp_strides)
                # conv_output.shape = [batch_size, X_img_h, X_img_w, # of channels]
                reshaped_conv_output = self.reshape_for_lstm(conv_output, cnfg.dim_time_after_conv, cnfg.dim_per_time_after_conv)
                
            with tf.variable_scope('lstm'):
                lstm_outputs = self.get_bidirectional_lstm_outputs(reshaped_conv_output, 
                                                                   dim_per_time=cnfg.dim_per_time_after_conv, 
                                                                   state_sizes=cnfg.lstm_state_sizes)
                X1 = tf.reshape(lstm_outputs, [-1, cnfg.n_flat])
            
            with tf.variable_scope('flat1'):
                X2 = self.apply_bn_dr_XWplusb(X=X1, W_shape=[cnfg.n_flat, cnfg.n_hidden])

            with tf.variable_scope('flat2'):
                self.logits = self.apply_bn_dr_XWplusb(X=X2, W_shape=[cnfg.n_hidden, cnfg.Y_vector_len], skip_relu=True)       
                
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
