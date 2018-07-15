from collections import namedtuple
import tensorflow as tf
import time
import os
import sys
import matplotlib.pyplot as plt
import ircnn_model
import glob as gb
import cv2
import numpy as np
import read_tfrecord

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('tfrecord_path', './data/data.tfrecords','')
tf.app.flags.DEFINE_string('test_set_path', './','')
tf.app.flags.DEFINE_string('mode', 'Eval', 'train or eval.')
tf.app.flags.DEFINE_string('eval_data_path', './Test/Set14',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_string('save_eval_path', 'result', '')
tf.app.flags.DEFINE_integer('max_iter', 160000, '')
tf.app.flags.DEFINE_bool('is_color', True, '')
tf.app.flags.DEFINE_integer('image_size', 64, 'Image side length.')
tf.app.flags.DEFINE_integer('batch_size', 128,'')
tf.app.flags.DEFINE_integer('sigma', 25,'')
tf.app.flags.DEFINE_string('log_dir', 'log_dir',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints','')



def train(hps):

    images, labels = read_tfrecord.build_input(
        FLAGS.tfrecord_path, FLAGS.image_size, FLAGS.batch_size, FLAGS.sigma, FLAGS.is_color)

    model = dncnn_model.IRCNN(hps, images, labels, FLAGS.mode)
    model.build_graph()
    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.
            TRAINABLE_VARS_PARAMS_STAT_OPTIONS)

    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': model.global_step,
                 'loss': model.cost},
        every_n_iter=10)

    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=FLAGS.log_dir,
        #summary_op=tf.summary.merge(model.summaries))
        summary_op=model.summaries
    )

    checkpoint_hook = tf.train.CheckpointSaverHook(
        checkpoint_dir=FLAGS.checkpoint_dir,
        save_steps=1000
    )

    class _LearningRateSetterHook(tf.train.SessionRunHook):
        """Sets learning_rate based on global step."""

        def begin(self):
            self._lrn_rate = 0.001

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(
                model.global_step,  # Asks for global step value.
                feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

        def after_run(self, run_context, run_values):
            train_step = run_values.results
            if train_step < 40000:
                self._lrn_rate = 0.001
            elif train_step < 80000:
                self._lrn_rate = 0.0001
            elif train_step < 120000:
                self._lrn_rate = 0.00001

           


    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.train.MonitoredTrainingSession(
            hooks=[logging_hook, _LearningRateSetterHook(), checkpoint_hook],
            chief_only_hooks=[summary_hook],
            save_summaries_steps=0,
            config=tf.ConfigProto(gpu_options=gpu_options)) as mon_sess:

        for i in range(FLAGS.max_iter):
            mon_sess.run(model.train_op)


def eval(hps):

    images = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    labels = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    model = ircnn_model.IRCNN(hps, images, labels, FLAGS.mode)
    model.build_graph()
    saver = tf.train.Saver()

    swts = ['.jpg', '*.png', '*.JPG', '*.bmp']
    path_lists = []
    for swt in swts:
        path_lists.extend(gb.glob(os.path.join(FLAGS.eval_data_path, swt)))
    try:
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    except tf.errors.OutOfRangeError as e:
        tf.logging.error('Cannot restore checkpoint: %s', e)

    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        tf.logging.info('No model to eval yet at %s', FLAGS.checkpoint_dir)
    zeros = np.zeros([1, 100, 100, 3])
    with tf.Session() as sess:
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        sess.run(model.clear, feed_dict={images:zeros, labels:zeros})
        for path in path_lists:
            gt = cv2.imread(path)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            gt = np.expand_dims(gt, axis=0)
            gt = gt.astype(np.float32) / 255.
            tf_psnr = tf.image.psnr(labels, model.clear, 1.)
            tf_ssim = tf.image.ssim(labels, model.clear, 1.)
            noisy = gt + FLAGS.sigma/255.*np.random.standard_normal(gt.shape)
            img, psnr, ssim = sess.run([model.clear, tf_psnr, tf_ssim], feed_dict={images:noisy, labels:gt})
            image_name = os.path.basename(path)
            print('%s, PSNR = %4.2f dB, SSIM = %4.4f'%(image_name, psnr[0], ssim[0]))
            img = img*255
            img[img<0] = 0
            img[img>255] = 255
            img = img.astype('uint8')
            noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min())
            noisy = noisy*255
            noisy[noisy<0] = 0
            noisy[noisy>255] == 255
            noisy = noisy.astype('uint8')
            img = np.concatenate([noisy,img], axis=2)
            cv2.imwrite(os.path.join(FLAGS.save_eval_path, image_name), cv2.cvtColor(np.squeeze(img), cv2.COLOR_RGB2BGR))




def main(_):
   
    hps = ircnn_model.HParams(batch_size=FLAGS.batch_size,
                               min_lrn_rate=0.00001,
                               lrn_rate=0.001,
                               num_conv=7,
                               weight_decay_rate=0.0001,
                               optimizer='adam')


    if FLAGS.mode == 'Train':
        train(hps)
    elif FLAGS.mode == 'Eval':
        eval(hps)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()


















