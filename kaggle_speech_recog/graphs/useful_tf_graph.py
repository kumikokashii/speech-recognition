    
import tensorflow as tf

from sklearn.utils import shuffle
from datetime import datetime, timedelta

import os
import shutil

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
        
        joined_name = '_'.join([self.name, self.g_cnfg.name, cnfg.name])
        self.make_ckp_tb_dir(cnfg.ckp_dir, cnfg.tb_dir, joined_name)
        self.log = Log(cnfg.log_dir, joined_name, self.name, self.ckp_dir, self.tb_dir, self.g_cnfg, cnfg)
        
        with tf.Session(graph=self) as self.sess: 
            # Initializations
            tf.global_variables_initializer().run()  # Graph variables
            self.writer = tf.summary.FileWriter(self.tb_dir, self.sess.graph)  # Tensorboard
            self.saver_hourly = tf.train.Saver(max_to_keep=None)  # Model saver for hourly models
            self.saver_best = tf.train.Saver()  # Model saver for best models

            # Training loop
            for step in range(0, cnfg.max_step):
                if step == 0:  # Only first time
                    self.log.train_start = datetime.now()
                    print('='*60)
                    print(joined_name)
                    print('='*60)
                    print('Training starts @ {:%m/%d/%Y %H:%M:%S}'.format(self.log.train_start))
                    
                    self.offset = 0
                    self.last_hr_model_time = datetime.now()
                    self.patient_till = float('inf')

                X_batch, Y_batch = self.get_next_batch()
                
                _, summary = self.sess.run([self.optimizer, self.summarizer], 
                                           feed_dict={self.X: X_batch, self.Y: Y_batch, 
                                                      self.keep_prob: cnfg.dropout_keep_prob, 
                                                      self.is_training: True})
                
                if step % cnfg.log_every == 0:  # Keep track of training progress                    
                    ll_train = self.logloss.eval(feed_dict={self.X: X_batch, self.Y: Y_batch, 
                                                            self.keep_prob: 1.0, self.is_training: False})
                    ll_valid = self.get_ll_valid()  # Split into batches to avoid running out of resource
                    
                    self.save_basics(step, ll_train, ll_valid, summary)
                    self.ave_ll_valid = self.log.ave_ll_valid[-1]
                    
                    self.make_ckp_if_hour_passed(step)
                    self.make_ckp_if_best(step)
                    
                    # Done if patience is over
                    if (step > cnfg.start_step_early_stopping) & (self.ave_ll_valid > self.patient_till):
                        print('Early stopping now')
                        break

                if (annotate) & (step % cnfg.print_every == 0):
                    print('Step {:,} ends @ {:%m/%d/%Y %H:%M:%S} [Train ll] {:.3f} [Ave valid ll] {:.3f}'.format(step, datetime.now(), ll_train, self.ave_ll_valid))

            # The End
            log.train_end = datetime.now()
            print('Training ends @ {:%m/%d/%Y %H:%M:%S}'.format(log.train_end))
            
            self.log.save()
            self.make_ckp(self.saver_hourly, 'hourly', step)
                 
    def load_and_predict(self, X_test, path2ckp):
        # User specifies checkpoint
        with tf.Session(graph=self) as sess:
            tf.global_variables_initializer().run()
            
            saver = tf.train.Saver()
            saver.restore(sess, path2ckp)  # Load model
            
            Y_test = self.logits.eval(feed_dict={self.X: X_test,  
                                                 self.keep_prob: 1.0,
                                                 self.is_training: False})
        return Y_test
            
    def predict(self, X_test, ckp_dir=None):
        # Use best model i.e. model with best ave ll valid
        if ckp_dir is None:
            ckp_dir = self.ckp_dir
        
        ckp = max(os.listdir(ckp_dir + '/best'), key=os.path.getctime)
        path2ckp = '/'.join([ckp_dir, 'best', ckp])
        
        return self.load_and_predict(path2ckp, X_test)

    
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
        
        if self.offset <= (self.len_X_train - self.batch_size):  # Enough for next batch
            self.offset += self.batch_size
        else:  # Reached epoch end
            self.offset = 0
        
        return X_batch, Y_batch

    def get_ll_valid(self):
        offset = 0
        sum_ll_valid = 0
        while (offset < self.len_X_valid):
            X_batch = self.X_valid[offset: offset+self.batch_size, :]
            Y_batch = self.Y_valid[offset: offset+self.batch_size, :]
            offset += self.batch_size
            
            batch_ll_valid = self.logloss_batch_sum.eval(feed_dict={self.X: X_batch, self.Y: Y_batch, 
                                                                    self.keep_prob: 1.0, self.is_training: False})
            sum_ll_valid += batch_ll_valid
            
        return (sum_ll_valid / self.len_X_valid)    
    
    def save_basics(self, step, ll_train, ll_valid, summary):
        self.log.record(step, ll_train, ll_valid)  # Log file
        self.log.save()
        self.writer.add_summary(summary, step)  # Tensorboard
        
    def make_ckp(self, saver, sub_dir, step):
        path_ckp = saver.save(self.sess, '/'.join([self.ckp_dir, sub_dir, 'model']), 
                              global_step=step, latest_filename=sub_dir+'_checkpoint') 
        return path_ckp
        
    def make_ckp_if_hour_passed(self, step):
        if datetime.now() <= self.last_hr_model_time + timedelta(hours=1):
            return
        
        path_ckp = self.make_ckp(self.saver_hourly, 'hourly', step)
        self.last_hr_model_time = datetime.now()
        
        if self.annotate:
            print('step {:,} Hourly model saved @ {:%m/%d/%Y %H:%M:%S}'.format(step, datetime.now()))

    def make_ckp_if_best(self, step):
        if self.ave_ll_valid > self.log.best_model_ll:
            return
        
        path_ckp = self.make_ckp(self.saver_best, 'best', step)
        self.patient_till = self.ave_ll_valid + self.early_stopping_patience
        self.log.update_best_model(self.patient_till)
        
        if self.annotate:
            print('Step {:,} Best model saved @ {:%m/%d/%Y %H:%M:%S} [Ave valid ll] {:.3f}'.format(step, datetime.now(), self.ave_ll_valid))
        
