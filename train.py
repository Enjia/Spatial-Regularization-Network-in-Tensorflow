import os
import sys
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
N_GPUs = 4
MAX_STEPS = 60000
BREIF_SUMMARY_PERIOD = 10
DETAILED_SUMMARY_PERIOD = 200
CHEKCPOINT_PERIOD = 5000
PROFILING = 21
OPTIMIZER = 'adam'
LOSS_TYPE = 'ycls_loss'
DATA_DIR = './'
DATA_NAME = 'multilabel_clothes_jpeg.tfrecords'
PROFILING_REPORT_PATH = './timeline.json'
TRAIN_MODE = ['pretrained_ImageNet', 'main_net_finetune', 'fatt_conv1_finetune', 'fsr_finetune', 'joint_finetune']
PRETRAINED_MODEL_DIR = './ckpt'

class SpatialRegularizationNetworkClassification(object):
    def __init__(self):
        self.classifier = SRNet
        with tf.device("/cpu:0"):
            self.global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int64)
            tf.summary.scalar("global_step", self.global_step, collections=["brief"])
        self.loss_type = LOSS_TYPE
        self._set_up_train_net_multigpu()
        self.should_stop = False

    def _tower_loss(self, train_batch, loss):
        images = train_batch['image_jepg']
        labels = train_batch['label']
        cls = self.classifier(images)
        # scores of different module
        if loss == "ycls_loss":
            ycls_raw, ycls = cls.main_net()
            ycls_loss = cls.build_loss(labels, ycls_raw, loss)
            return ycls_loss
        elif loss == "yatt_loss":
            y_att_raw_score, y_att_score, _ = cls.srnet()
            yatt_loss = cls.build_loss(labels, y_att_raw_score, loss)
            return yatt_loss
        elif loss == "ysr_loss":
            _, _, ysr_raw_score = cls.srnet()
            ysr_loss = cls.build_loss(labels, ysr_raw_score, loss)
            return ysr_loss
        else:
            final_score = cls.final_score()
            final_loss = cls.build_loss(labels, final_score, loss)
            return final_loss

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

            tower_grads = []
            tower_losses = []
            for i in range(N_GPUs):
                logging.info("Setting up tower %d" % i)
                with tf.device("/gpu:%d" % i):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=(i > 0)):
                        with tf.name_scope("tower_%d" % i):
                            loss = self._tower_loss(sub_batches[i], LOSS_TYPE)
                            grads = optimizer.compute_gradients(loss)
                            tower_grads.append(grads)
                            tower_losses.append(loss)
            self.loss = tf.add_n([tower_losses])
            tf.summary.scalar("total loss", self.loss, collections=['brief'])
            with tf.name_scope("average_loss"):
                grads = self._average_gradients(tower_grads)
            with tf.variable_scope("optimizer"):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)

            for var in tf.all_variables():
                summary_name = 'parameters/' + var.name.split(':')[0]
                tf.summary.histogram(summary_name, var, collections=['detailed'])
            self.brief_summary_op = tf.summary.merge_all(key='brief')
            self.detailed_summary_op = tf.summary.merge_all(key="detailed")

    def train_and_eval(self, train_mode_index):
        # train_mode_index is limited from: 1, 2, 3, represent each mode in TRAIN_MODE except pretrained ImageNet
        sess_config = tf.ConfigProto(log_device_placement=True,
                                     allow_soft_placement=True)
        with tf.Session(config=sess_config) as sess:
            summary_writer = tf.summary.FileWriter("./graphs", graph=sess.graph)
            saver = tf.train.Saver(max_to_keep=20)
            train_mode = TRAIN_MODE[train_mode_index-1]
            ckpt_path = PRETRAINED_MODEL_DIR + train_mode + '.ckpt'
            model_loader = tf.train.Saver()
            model_loader.restore(sess, ckpt_path)
            tf.assign(self.global_step, 0).eval()
            logging.info('Resuming checkpoint %s' % ckpt_path)
            ckpt_save_path = PRETRAINED_MODEL_DIR + "/" + TRAIN_MODE[train_mode_index] + ".ckpt"

            with slim.queues.QueueRunners(sess):
                logging.info("Training loop started")
                start_step = self.global_step.eval()
                tqdm_range = trange(start_step, MAX_STEPS)
                for step in tqdm_range:
                    need_brief_summary = step % BREIF_SUMMARY_PERIOD == 0
                    need_detailed_summary = step % DETAILED_SUMMARY_PERIOD == 0
                    need_save_checkpoint = step > 0 and step % CHEKCPOINT_PERIOD == 0
                    need_profiling = PROFILING and step == PROFILING

                    train_fetches = {}
                    train_fetches['train_op'] = self.train_op
                    train_fetches['loss'] = self.loss
                    train_fetches['lr'] = self.current_lr
                    train_fetches['brief_summary'] = self.brief_summary_op
                    if need_brief_summary:
                        train_fetches['loss'] = self.loss
                        train_fetches['lr'] = self.current_lr
                        train_fetches['brief_summary'] = self.brief_summary_op
                    if need_detailed_summary:
                        train_fetches['detailed_summary'] = self.detailed_summary_op
                    if need_profiling:
                        run_metadata = tf.RunMetadata()
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    else:
                        run_metadata = None
                        run_options = None

                    time_start = time.time()
                    outputs = sess.run(train_fetches, options=run_options, run_metadata=run_metadata)
                    if need_brief_summary:
                        summary_str = outputs['brief_summary']
                        summary_writer.add_summary(summary_str, step)
                        loss = outputs['loss']
                        lr = outputs['lr']
                        loss_str = '%.2f' % loss
                        lr_str = '%.2e' % lr
                        tqdm_range.set_postfix(loss=loss_str, lr=lr_str)

                        if np.isnan(loss) or loss > 1e3:
                            logging.critical('Training diverges. Terminating.')
                    if need_detailed_summary:
                        summary_str = outputs['detailed_summary']
                        summary_str.add_summary(summary_str, step)
                    if need_profiling:
                        tl = timeline.Timeline(run_metadata.step_stats)
                        ctf = tl.generate_chrome_trace_format()
                        tl_write_path = PROFILING_REPORT_PATH
                        with open(tl_write_path, 'w') as f:
                            f.write(ctf)
                        logging.info('Profiling results written to {}'.format(tl_write_path))
                        summary_writer.add_summary(run_metadata, "step_%d" % step)
                        logging.info('Profiling results written')
                    if need_save_checkpoint:
                        saver.save(sess, ckpt_save_path, global_step=self.global_step)
                        logging.info('Checkpoint saved to %s' % ckpt_save_path)
                    if self.should_stop == True:
                        break

                logging.info('Training loop ended')
                saver.save(sess, ckpt_save_path, global_step=self.global_step.eval())
                logging.info('Checkpoint saved to %s' % ckpt_save_path)

if __name__ == '__main__':
    SRNClassifier = SpatialRegularizationNetworkClassification()
    SRNClassifier.train_and_eval()
