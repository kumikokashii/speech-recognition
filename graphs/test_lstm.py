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
            self.y = tf.placeholder(tf.float32, [None, cnfg.y_vector_len])

            W1 = self.weight_variable([cnfg.X_vector_len, cnfg.n_hidden], 0.015, 'W1')
            b1 = self.bias_variable(0.1, [cnfg.n_hidden], 'b1')
            Xw1 = tf.matmul(self.X, W1) + b1
            X2 = tf.nn.relu(Xw1)

            W2 = self.weight_variable([cnfg.n_hidden, cnfg.y_vector_len], 0.015, 'W2')
            b2 = self.bias_variable(0.1, [cnfg.y_vector_len], 'b2')
            self.logits = tf.matmul(X2, W2) + b2

            self.logloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits))

            learning_rate = tf.train.exponential_decay(cnfg.lr_initial, global_step, 
                                                       cnfg.lr_decay_steps, cnfg.lr_decay_rate, 
                                                       staircase=True)

            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.logloss, global_step=global_step)

            tf.summary.scalar('logloss', self.logloss)
            tf.summary.scalar('learning_rate', learning_rate)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

            self.summarizer = tf.summary.merge_all()
