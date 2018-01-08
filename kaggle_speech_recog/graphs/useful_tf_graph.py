import tensorflow as tf

from sklearn.utils import shuffle
from datetime import datetime, timedelta

import os
import shutil
import numpy as np

from .train_log import *


class UsefulTFGraph(tf.Graph):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__

    def train_model(self, cnfg, XY_train_valid, annotate=True):
        # Prep
        self.batch_size = cnfg.batch_size
        self.random_pick_unknown = getattr(cnfg, 'random_pick_unknown', False)
        self.early_stopping_patience = cnfg.early_stopping_patience
        self.annotate = annotate

        # Break up data
        if self.random_pick_unknown:
            self.X_train_known, self.X_train_unknown, self.Y_train_known, self.Y_train_unknown, self.X_valid, self.Y_valid = XY_train_valid
        else:
            self.X_train, self.Y_train, self.X_valid, self.Y_valid = XY_train_valid
            
        # More prep
        if self.random_pick_unknown:
            self.len_X_train = int(len(self.X_train_known) / 11 * 12)
            self.n_random_unknown = self.len_X_train - len(self.X_train_known)
        else:
            self.len_X_train = len(self.X_train)
        self.len_X_valid = len(self.X_valid)        
        
        # For checkpoint models and log
        joined_name = '_'.join([self.name, self.cnfg.name, cnfg.name])
        self.make_ckp_tb_dir(cnfg.ckp_dir, cnfg.tb_dir, joined_name)
        self.log = Log(cnfg.log_dir, joined_name, self.name, self.ckp_dir, self.tb_dir, self.cnfg, cnfg)
        
        with tf.Session(graph=self) as self.sess: 
            # Initializations
            tf.global_variables_initializer().run()  # Graph variables
            self.writer = tf.summary.FileWriter(self.tb_dir, self.sess.graph)  # Tensorboard
            self.saver_hourly = tf.train.Saver(max_to_keep=None)  # Model saver for hourly models
            self.saver_best = tf.train.Saver()  # Model saver for best models

            # Training loop
            for step in range(1, cnfg.max_step):
                if step == 1:  # Only first time
                    self.log.train_start = datetime.now()
                    print('='*60)
                    print(joined_name)
                    print('='*60)
                    print('Training starts @ {:%m/%d/%Y %H:%M:%S}'.format(self.log.train_start))
                    
                    self.offset = 0
                    epoch = 0
                    self.last_hr_model_time = datetime.now()
                    self.patient_till = float('inf')

                if self.offset == 0:
                    epoch += 1
                X_batch, Y_batch = self.get_next_batch()  # self.offset gets incremented
                if hasattr(cnfg, 'add_noise'):
                    X_batch = self.add_noise(X_batch, *cnfg.add_noise)
                
                _, summary = self.sess.run([self.optimizer, self.summarizer], 
                                           feed_dict={self.X: X_batch, self.Y: Y_batch, 
                                                      self.keep_prob: cnfg.dropout_keep_prob, 
                                                      self.is_training: True})
                
                if (step == 1) or (self.offset == 0) or (step % cnfg.log_every == 0):  # Keep track of training progress
                    accu_train, ll_train = self.sess.run([self.accuracy, self.logloss], 
                                                         feed_dict={self.X: X_batch, self.Y: Y_batch, 
                                                                    self.keep_prob: 1.0, self.is_training: False})
                    accu_valid, ll_valid = self.get_accu_ll_valid()  # Split into batches to avoid running out of resource
                    
                    self.save_basics(step, epoch, accu_train, ll_train, accu_valid, ll_valid, summary)
                    self.ave_ll_valid = self.log.ave_ll_valid[-1]
                    
                    self.make_ckp_if_hour_passed(epoch, step)
                    self.make_ckp_if_best(epoch, step)
                    
                    # Done if patience is over
                    if (step > cnfg.start_step_early_stopping) & (self.ave_ll_valid > self.patient_till):
                        print('Early stopping now')
                        break

                if (annotate) and ((step == 1) or (step % cnfg.print_every == 0)):
                    print('Epoch {:,} Step {:,} ends @ {:%m/%d/%Y %H:%M:%S} [Train] {:.3f}, {:.1f}% [Valid] {:.1f}% [Ave valid] {:.3f}'.format(epoch, step, datetime.now(), ll_train, accu_train*100, accu_valid*100, self.ave_ll_valid))

            # The End
            log.train_end = datetime.now()
            print('Training ends @ {:%m/%d/%Y %H:%M:%S}'.format(log.train_end))
            
            self.log.save()
            self.make_ckp(self.saver_hourly, 'hourly', step)
                 
    def load_and_predict(self, X_test, path2ckp, batch_size, annotate=True):
        len_X_test = len(X_test)
        
        # User specifies checkpoint
        with tf.Session(graph=self) as sess:
            tf.global_variables_initializer().run()
            
            saver = tf.train.Saver()
            saver.restore(sess, path2ckp)  # Load model

            Y_test = np.empty([len_X_test, self.cnfg.Y_vector_len])
            offset = 0
            done_check = 15000
            
            if annotate:
                print('Predicting starts @ {:%m/%d/%Y %H:%M:%S}'.format(datetime.now()))
            while (offset < len_X_test):
                X_batch = X_test[offset: offset+batch_size, :]
                Y_batch = self.logits.eval(feed_dict={self.X: X_batch, 
                                                      self.keep_prob: 1.0,
                                                      self.is_training: False})
                Y_test[offset: offset+batch_size, :] = Y_batch
                
                offset += batch_size
                if done_check <= offset:
                    if annotate:
                        print('{:,} datapoints completed at {:%m/%d/%Y %H:%M:%S}'.format(offset, datetime.now()))
                    done_check += 15000  
            if annotate:
                print('Predicting ends @ {:%m/%d/%Y %H:%M:%S}'.format(datetime.now()))

        return Y_test
            
    def predict(self, X_test, ckp_dir=None, batch_size=10000, annotate=True):
        # Use best model i.e. model with best ave ll valid
        if ckp_dir is None:
            ckp_dir = self.ckp_dir
        
        path2ckp = tf.train.latest_checkpoint(ckp_dir + '/best', 'best_checkpoint')
        
        return self.load_and_predict(X_test, path2ckp, batch_size, annotate)     

    def get_sigmoid(self, X_test, ckp_dir, batch_size=10000, annotate=True):
        len_X_test = len(X_test)
        path2ckp = tf.train.latest_checkpoint(ckp_dir + '/best', 'best_checkpoint')
        
        with tf.Session(graph=self) as sess:
            tf.global_variables_initializer().run()
            
            saver = tf.train.Saver()
            saver.restore(sess, path2ckp)  # Load model

            sigmoid_test = np.empty([len_X_test, 1])
            offset = 0
            done_check = 15000
            
            if annotate:
                print('Getting sigmoid starts @ {:%m/%d/%Y %H:%M:%S}'.format(datetime.now()))
            while (offset < len_X_test):
                X_batch = X_test[offset: offset+batch_size, :]
                sigmoid_batch = self.sigmoid.eval(feed_dict={self.X: X_batch, 
                                                             self.keep_prob: 1.0,
                                                             self.is_training: False})
                sigmoid_test[offset: offset+batch_size, :] = sigmoid_batch
                
                offset += batch_size
                if done_check <= offset:
                    if annotate:
                        print('{:,} datapoints completed at {:%m/%d/%Y %H:%M:%S}'.format(offset, datetime.now()))
                    done_check += 15000  
            if annotate:
                print('Getting sigmoid ends @ {:%m/%d/%Y %H:%M:%S}'.format(datetime.now()))

        return sigmoid_test        
    
    # Helpers for train_model       
    def make_ckp_tb_dir(self, ckp_dir, tb_dir, joined_name):
        # Make tensorboard and checkpoint directories
        self.ckp_dir = ckp_dir + '/' + joined_name
        self.tb_dir = tb_dir + '/' + joined_name
        
        for dir_ in [self.tb_dir, self.ckp_dir]:
            if os.path.isdir(dir_):
                shutil.rmtree(dir_)
            os.makedirs(dir_)

        # Sub directories for checkpoint
        os.makedirs(self.ckp_dir + '/hourly')
        os.makedirs(self.ckp_dir + '/best')
        
    def get_next_batch(self):
        if self.random_pick_unknown:
            i_random_unknown = np.random.choice(len(self.X_train_unknown), size=self.n_random_unknown, replace=False)
            self.X_train = np.concatenate((self.X_train_known, self.X_train_unknown[i_random_unknown, :]))
            self.Y_train = np.concatenate((self.Y_train_known, self.Y_train_unknown[i_random_unknown, :]))           
        
        if self.offset == 0:  # Shuffle every epoch
            self.X_train, self.Y_train = shuffle(self.X_train, self.Y_train)

        X_batch = self.X_train[self.offset: self.offset+self.batch_size, :]
        Y_batch = self.Y_train[self.offset: self.offset+self.batch_size, :]
        
        self.offset += self.batch_size  # For next round
        if self.offset >= self.len_X_train:
            self.offset = 0
        
        return X_batch, Y_batch
    
    def add_noise(self, X, noise_p, noise_X_list):
        X = np.copy(X)
        n = int(len(X)*noise_p)
        
        i_X_list = np.random.choice(len(X), n, replace=False)
        i_noise_X_list = np.random.choice(len(noise_X_list), n)
        
        for j in range(n):
            i_X = i_X_list[j]
            i_noise_X = i_noise_X_list[j]
            X[i_X] = X[i_X] + noise_X_list[i_noise_X]

        return X

    def get_accu_ll_valid(self):
        offset = 0
        count_accu_valid = 0
        sum_ll_valid = 0
        while (offset < self.len_X_valid):
            X_batch = self.X_valid[offset: offset+self.batch_size, :]
            Y_batch = self.Y_valid[offset: offset+self.batch_size, :]
            offset += self.batch_size        
            
            batch_accu_valid, batch_ll_valid = self.sess.run([self.accuracy_batch_count, self.logloss_batch_sum],
                                                             feed_dict={self.X: X_batch, self.Y: Y_batch, 
                                                                        self.keep_prob: 1.0, self.is_training: False})
            count_accu_valid += batch_accu_valid
            sum_ll_valid += batch_ll_valid
            
        accu_valid = count_accu_valid / self.len_X_valid
        ll_valid = sum_ll_valid / self.len_X_valid
        return accu_valid, ll_valid
    
    def save_basics(self, step, epoch, accu_train, ll_train, accu_valid, ll_valid, summary):
        self.log.record(step, epoch, accu_train, ll_train, accu_valid, ll_valid)  # Log file
        self.log.save()
        self.writer.add_summary(summary, step)  # Tensorboard
        
    def make_ckp(self, saver, sub_dir, step):
        path_ckp = saver.save(self.sess, '/'.join([self.ckp_dir, sub_dir, 'model']), 
                              global_step=step, latest_filename='_'.join([sub_dir, 'checkpoint'])) 
        return path_ckp
        
    def make_ckp_if_hour_passed(self, epoch, step):
        if datetime.now() <= self.last_hr_model_time + timedelta(hours=1):
            return
        
        path_ckp = self.make_ckp(self.saver_hourly, 'hourly', step)
        self.last_hr_model_time = datetime.now()
        
        if self.annotate:
            print('Epoch {:,} Step {:,} Hourly model saved @ {:%m/%d/%Y %H:%M:%S}'.format(epoch, step, datetime.now()))

    def make_ckp_if_best(self, epoch, step):
        if self.ave_ll_valid > self.log.best_model_ll:
            return
        
        path_ckp = self.make_ckp(self.saver_best, 'best', step)
        self.patient_till = self.ave_ll_valid + self.early_stopping_patience
        self.log.update_best_model(self.patient_till)
        
        if self.annotate:
            print('Epoch {:,} Step {:,} Best model saved @ {:%m/%d/%Y %H:%M:%S} [Ave valid] {:.3f}'.format(epoch, step, datetime.now(), self.ave_ll_valid))
        
