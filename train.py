import os
import sys
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
import logging
import numpy as np
from tqdm import trange

from network import SRNet
from data_reader import Reader

LR_POLICY = 'staircase'
LR_BREAKPOINT = [10000, 20000, 60000, 75000, 90000]
LR_DECAY = [0.64, 0.8, 1.0, 0.1, 0.01]
BASE_LR = 5e-4
BATCH_SIZE = 32
DATA_DIR = "./"
DATA_NAME = "multilabel_clothes_jpeg.tfrecords"
N_GPUs = 4
OPTIMIZER = 'adam'

class SpatialRegularizationNetworkClassification(object):
    def __init__(self):
        self.classifier = SRNet
        with tf.device("/cpu:0"):
            self.global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int64)
            tf.summary.scalar("global_step", self.global_step, collections=["brief"])

        self._set_up_train_net_multigpu()
        self.should_stop = False

    def _tower_loss(self, train_batch):
        images = train_batch['image_jepg']
        cls = self.classifier(images)
        # scores of different module
        ycls_raw, ycls = cls.main_net()
        y_att_raw_score, y_att_score, y_sr_raw_score = cls.srnet()
        final_score = cls.final_score()

        tower_loss = cls.build_loss()

    def _average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)

            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def _set_up_train_net_multigpu(self):
        with tf.device("/cpu:0"):
            # learning rate decay
            with tf.name_scope("lr_decay"):
                if LR_POLICY == 'staircase':
                    lr_breakpoints = [int(lbp) for lbp in LR_BREAKPOINT]
                    lr_decays = [int(ld) for ld in LR_DECAY]
                    assert len(lr_breakpoints) == len(lr_decays)
                    pred_fn_pairs = []
                    for lr_decay, lr_breakpoint in zip(lr_decays, lr_breakpoints):
                        fn = (lambda o: lambda: tf.constant(o, tf.float32))(lr_decay)
                        pred_fn_pairs.append((tf.less(self.global_step, lr_breakpoint), fn))
                    lr_decay = tf.case(pred_fn_pairs, default=(lambda: tf.constant(1.0)))
                else:
                    logging.error("Unknown lr_policy: {}".format(LR_POLICY))
                    sys.exit(1)
                self.current_lr = lr_decay * BASE_LR
                tf.summary.scalar('lr', self.current_lr, collections=["brief"])

            # input data
            with tf.name_scope("input_data"):
                batch_size = BATCH_SIZE
                train_data_list = os.path.join(DATA_DIR, DATA_NAME)
                train_reader = Reader(train_data_list, is_training=True)
                train_batch = train_reader.dequeue(batch_size)
                sub_batch_size = int(batch_size / N_GPUs)
                logging.info('Batch size is {} on each of the {} GPUs'.format(sub_batch_size, N_GPUs))
                sub_batches = []
                for i in range(N_GPUs):
                    sub_batch = {}
                    for k, v in train_batch.items():
                        sub_batch[k] = v[i*sub_batch_size: (i+1)*sub_batch_size]
                    sub_batches.append(sub_batch)

            if OPTIMIZER == 'sgd':
                optimizer = tf.train.MomentumOptimizer(self.current_lr, 0.9)
                logging.info('Using SGD optimizer. Momentum={}'.format(0.9))
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(self.current_lr)
                logging.info('Using ADAM optimizer.')
            elif OPTIMIZER == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.current_lr)
                logging.info('Using RMSProp optimizer.')
            else:
                logging.critical('Unsupported optimizer {}'.format(OPTIMIZER))
                sys.exit(1)

            return sub_batches, optimizer
