    
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
        self.early_stopping_patience = cnfg.early_stopping_patience

        self.X_train, self.Y_train, self.X_valid, self.Y_valid = XY_train_valid
        self.len_X_train = len(self.X_train)
        self.len_X_valid = len(self.X_valid)
        
        self.annotate = annotate
        
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
                 
    def load_and_predict(self, X_test, path2ckp, batch_size):
        len_X_test = len(X_test)
        
        # User specifies checkpoint
        with tf.Session(graph=self) as sess:
            tf.global_variables_initializer().run()
            
            saver = tf.train.Saver()
            saver.restore(sess, path2ckp)  # Load model

            Y_test = np.empty([len_X_test, self.cnfg.Y_vector_len])
            offset = 0
            done_check = 15000
            
            print('Predicting starts @ {:%m/%d/%Y %H:%M:%S}'.format(datetime.now()))
            while (offset < len_X_test):
                X_batch = X_test[offset: offset+batch_size, :]
                Y_batch = self.logits.eval(feed_dict={self.X: X_batch, 
                                                      self.keep_prob: 1.0,
                                                      self.is_training: False})
                Y_test[offset: offset+batch_size, :] = Y_batch
                
                offset += batch_size
                if done_check <= offset:
                    print('{:,} datapoints completed at {:%m/%d/%Y %H:%M:%S}'.format(offset, datetime.now()))
                    done_check += 15000             
            print('Predicting ends @ {:%m/%d/%Y %H:%M:%S}'.format(datetime.now()))

        return Y_test
            
    def predict(self, X_test, ckp_dir=None, batch_size=10000):
        # Use best model i.e. model with best ave ll valid
        if ckp_dir is None:
            ckp_dir = self.ckp_dir
        
        path2ckp = tf.train.latest_checkpoint(ckp_dir + '/best', 'best_checkpoint')
        
        return self.load_and_predict(X_test, path2ckp, batch_size)

    
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
        if self.offset == 0:  # Shuffle every epoch
            self.X_train, self.Y_train = shuffle(self.X_train, self.Y_train)

        X_batch = self.X_train[self.offset: self.offset+self.batch_size, :]
        Y_batch = self.Y_train[self.offset: self.offset+self.batch_size, :]
        
        self.offset += self.batch_size  # For next round
        if self.offset >= self.len_X_train:
            self.offset = 0
        
        return X_batch, Y_batch

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
        
