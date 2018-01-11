from .useful_tf_graph import *
import tensorflow as tf

class SpectrogramMultiLSTMRandomInputModify2(UsefulTFGraph):
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
    
    def get_weight_tensor(self, shape, stddev=0.015):
        return tf.get_variable(name='W', shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    
    def get_bias_tensor(self, shape, value=0.1):
        return tf.get_variable(name='b', shape=shape, initializer=tf.constant_initializer(value=value))

    def apply_batch_normalize(self, inputs):
        return tf.contrib.layers.batch_norm(inputs=inputs,
                                            updates_collections=None,
                                            is_training=self.is_training,
                                            scope='bn')

    def apply_convolution(self, inputs, n_filters, kernel_size, strides=(1, 1), per_channel=False):
        if per_channel:
            inputs_ = tf.expand_dims(inputs, -1)
        else:
            inputs_ = inputs
        
        conv_ = tf.layers.conv2d(inputs=inputs_,
                                 filters=n_filters,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding='SAME',
                                 name='conv')
        
        if per_channel:
            return tf.squeeze(conv_, [3])
        else:
            return conv_
        
    def apply_max_pooling(self, inputs, mp_size, strides):
        return tf.layers.max_pooling2d(inputs=inputs,
                                       pool_size=mp_size,
                                       strides=strides,
                                       padding='SAME',
                                       name='mp')

    def apply_ave_pooling(self, inputs, window, strides):
        return tf.nn.avg_pool(inputs, 
                              ksize=[1, window[0], window[1], 1], 
                              strides=[1, strides[0], strides[1], 1], 
                              padding='SAME', 
                              name='ave_pool')   
    
    def apply_bn_dr_conv_mp(self, inputs, keep_prob, conv_n_filters, conv_kernel_size, mp_size, mp_strides, skip_relu=False):
        bn_ = self.apply_batch_normalize(inputs)
        dr_ = tf.cond(self.is_training, 
                      lambda: tf.nn.dropout(bn_, keep_prob=keep_prob), 
                      lambda: tf.identity(bn_))        
        conv_ = self.apply_convolution(dr_, conv_n_filters, conv_kernel_size)
        if (mp_size == (1, 1)) and (mp_strides == (1, 1)):
            mp_ = conv_
        else:
            mp_ = self.apply_max_pooling(conv_, mp_size, mp_strides)
        
        if skip_relu:
            return mp_
        else:
            return tf.nn.relu(mp_)
                
    def apply_dr_ds_conv(self, inputs, keep_prob, kernel_size, strides, n_filters, end_with_relu=False):
        with tf.variable_scope('depthwise'):            
            # BN + Dropout + Relu
            inputs_ = self.apply_batch_normalize(inputs)
            inputs_ = tf.cond(self.is_training, 
                              lambda: tf.nn.dropout(inputs_, keep_prob=keep_prob), 
                              lambda: tf.identity(inputs_))
            inputs_ = tf.nn.relu(inputs_)                

            # Depthwise conv
            transpose_ = tf.transpose(inputs_, [3, 0, 1, 2])  # Channels first
            transpose_ = tf.map_fn(lambda x: self.apply_convolution(x, 1, kernel_size, strides, per_channel=True), transpose_)
            dw_ = tf.transpose(transpose_, [1, 2, 3, 0])  # Batch first

        with tf.variable_scope('pointwise'):
            # BN + Relu
            dw_ = self.apply_batch_normalize(dw_)
            dw_ = tf.nn.relu(dw_)

            # Pointwise conv
            pw_ = self.apply_convolution(dw_, n_filters, kernel_size=(1, 1))
            
        if end_with_relu:
            return tf.nn.relu(pw_)
        return pw_
    
    def reduce_conv_dim_for_lstm(self, inputs, n_time, n_per_time, n_after_per_time):
        # Flip height and width and reshape
        reshaped_ = tf.reshape(tf.transpose(inputs, [0, 2, 1, 3]), [-1, n_per_time])
        # Rows are batch1time1, batch1time2, ..., batch2time1, batch2time2, ...

        # Reduce
        reduced_ = self.apply_bn_dr_XWplusb(reshaped_, [n_per_time, n_after_per_time])

        # Reshape
        return tf.reshape(reduced_, [-1, n_time, n_after_per_time])      
    
    def reshape_for_lstm(self, inputs, n_time, n_per_time):
        # From conv output to lstm input
        temp = tf.transpose(inputs, [0, 2, 1, 3])  # Flip height and width  
        return tf.reshape(temp, [-1, n_time, n_per_time]) # Collapse channels
                
    def get_lstm_outputs(self, inputs, dim_per_time, state_sizes, keep_prob):
        # inputs.shape = (batch size) x (length of time) x (dim of data at each time)
        batch_size = tf.shape(inputs)[0]
        
        bn_ = self.apply_batch_normalize(inputs)

        cell_list = []
        initial_state_list = []
        input_sizes = [dim_per_time] + state_sizes
        for i in range(len(state_sizes)):
            cell = tf.contrib.rnn.BasicLSTMCell(state_sizes[i])  # forget_bias=1.0
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_size=input_sizes[i], dtype=tf.float32,
                                                 input_keep_prob=tf.cond(self.is_training, lambda: keep_prob, lambda: 1.0),
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

    def get_bidirectional_lstm_outputs(self, inputs, dim_per_time, state_sizes, keep_prob):
        # inputs.shape = (batch size) x (length of time) x (dim of data at each time)
        batch_size = tf.shape(inputs)[0]
        
        bn_ = self.apply_batch_normalize(inputs)

        cell_list = []
        initial_state_list = []
        input_sizes = [dim_per_time] + state_sizes
        for i in range(len(state_sizes)):
            cell = tf.contrib.rnn.BasicLSTMCell(state_sizes[i])  # forget_bias=1.0
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_size=input_sizes[i], dtype=tf.float32,
                                                 input_keep_prob=tf.cond(self.is_training, lambda: keep_prob, lambda: 1.0),
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
    
    def flatten_last_output(self, lstm_outputs, n_flat):  # For LSTM                    
        return tf.reshape(lstm_outputs[:, -1, :], [-1, n_flat])
    
    def flatten_maxpool_output(self, lstm_outputs, n, n_flat):  # For LSTM
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

    def apply_flat_layers(self, inputs, n_in, hiddens, n_out):
        n_list = [n_in] + hiddens
        len_n_list = len(n_list)
        Xi = inputs

        for i in range(1, len_n_list):
            with tf.variable_scope('flat{}'.format(i)):
                Xi = self.apply_bn_dr_XWplusb(X=Xi, W_shape=[n_list[i-1], n_list[i]])

        with tf.variable_scope('flat{}'.format(len_n_list)):
            Xlast = self.apply_bn_dr_XWplusb(X=Xi, W_shape=[n_list[-1], n_out], skip_relu=True)

        return Xlast        
 
    def prebuild(self, cnfg):
        cnfg.random_modify_args = (cnfg.X_img_h, cnfg.X_img_w,
                                   cnfg.random_modify_brightness_max_delta, 
                                   cnfg.random_modify_contrast_lower, cnfg.random_modify_contrast_upper)
        
        cnfg.n_flat = cnfg.lstm_state_sizes[-1]
        
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
            
            with tf.variable_scope('lstm'):
                lstm_outputs = self.get_lstm_outputs(tf.transpose(self.X, [0, 2, 1]), 
                                                     cnfg.X_img_h, cnfg.lstm_state_sizes, cnfg.lstm_keep_prob)

            with tf.variable_scope('flat'):
                Xflat = tf.reshape(lstm_outputs[:, -1, :], [-1, cnfg.n_flat])
                self.logits = self.apply_flat_layers(Xflat, cnfg.n_flat, cnfg.flat_hiddens, cnfg.Y_vector_len)                

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
