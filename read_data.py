import numpy as np
import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt

#读取制作好的tfrecords数据，返回数据和对应的标签

def read_and_decode(filename,batch_size): # read train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])# create a queue

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#return image and label

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.cast(img, tf.float32)*(1./255) - 0.5
    img = tf.reshape(img, [112, 112, 3])
    label = tf.cast(features['label'], tf.int32) #throw label tensor



    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size= batch_size,
                                                    num_threads=64,
                                                    capacity=2000,
                                                    min_after_dequeue=1500,
                                                    )
    return img_batch, tf.reshape(label_batch,[batch_size])


