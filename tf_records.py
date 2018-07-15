from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def image_to_tfexample(image_data):

    return tf.train.Example(features=tf.train.Features(feature={
      'image_raw': bytes_feature(image_data),
    }))


def creat_tfrecord(path, tf_filename):

    path_lists = tf.gfile.Glob(os.path.join(path, '*.jpg'))


    with tf.python_io.TFRecordWriter(tf_filename) as to_write:

        for path in path_lists:
            print('load %s'%path)
            with tf.gfile.FastGFile(path, 'rb') as to_read:

                image_string = to_read.read()

            example = image_to_tfexample(image_string)

            to_write.write(example.SerializeToString())

    print('Finish!')

if __name__ == '__main__':

    creat_tfrecord('./BSDS300', './data.tfrecords')
