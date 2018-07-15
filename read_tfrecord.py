from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np


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



def build_input(tfrecord_name, image_size, batch_size,  sigma=25, is_color = True):

    filename_queue = tf.train.string_input_producer([tfrecord_name])

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={'image_raw':tf.FixedLenFeature([], tf.string),})

    image = tf.image.decode_jpeg(features['image_raw'])

    if is_color:
        depth = 3
    else:
        depth = 1
    subimg = tf.random_crop(image, [image_size, image_size, depth])
    subimg = tf.image.convert_image_dtype(subimg, tf.float32)

    input_image = subimg + float(sigma)/255.*tf.random_normal(tf.shape(subimg))
    label = input_image - subimg

    example_queue = tf.RandomShuffleQueue(
        capacity=16 * batch_size,
        min_after_dequeue=8 * batch_size,
        dtypes=[tf.float32, tf.float32],
        shapes=[[image_size, image_size, depth], [image_size, image_size, depth]])

    num_threads = 16

    example_enqueue_op = example_queue.enqueue([input_image, label])

    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, [example_enqueue_op] * num_threads))

    images, labels = example_queue.dequeue_many(batch_size)

    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == depth
    assert len(labels.get_shape()) == 4
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[-1] == depth


    return images, labels