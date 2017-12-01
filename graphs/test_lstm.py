from .useful_tf_graph import *
import tensorflow as tf

class TestLSTM(UsefulTFGraph):
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
        self.g_cnfg = cnfg
        with self.as_default():
            global_step = tf.Variable(0, trainable=False)
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)

            self.X = tf.placeholder(tf.float32, [None, cnfg.X_vector_len])
            self.Y = tf.placeholder(tf.float32, [None, cnfg.Y_vector_len])
            
            X_lstm = tf.reshape(self.X, [-1, cnfg.X_vector_len, 1])
            # (batch size) x (length of time) x (dim of data at each time)
            
            batch_size = tf.shape(self.X)[0]
            initial_state = cell.zero_state(batch_size, tf.float32)
            cell = tf.contrib.rnn.BasicLSTMCell(cnfg.lstm_state_size)  # forget_bias=1.0
            outputs, final_state = tf.nn.dynamic_rnn(cell, X_lstm, initial_state=initial_state)
            # outputs is (batch size) x (length of time) x (lstm state size)
            final_output = outputs[:, -1, :]
            
            X1 = tf.reshape(final_output, [-1, cnfg.lstm_state_size])
            W1 = self.weight_variable([cnfg.lstm_state_size, cnfg.n_hidden], 0.015, 'W1')
            b1 = self.bias_variable(0.1, [cnfg.n_hidden], 'b1')
            XW1 = tf.matmul(X1, W1) + b1
            X2 = tf.nn.relu(XW1)

            W2 = self.weight_variable([cnfg.n_hidden, cnfg.y_vector_len], 0.015, 'W2')
            b2 = self.bias_variable(0.1, [cnfg.y_vector_len], 'b2')
            self.logits = tf.matmul(X2, W2) + b2

            self.logloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.logits))

            learning_rate = tf.train.exponential_decay(cnfg.lr_initial, global_step, 
                                                       cnfg.lr_decay_steps, cnfg.lr_decay_rate, 
                                                       staircase=True)

            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.logloss, global_step=global_step)

            tf.summary.scalar('logloss', self.logloss)
            tf.summary.scalar('learning_rate', learning_rate)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

            self.summarizer = tf.summary.merge_all()
