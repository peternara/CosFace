# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

###制作tfrecord数据的py，只需要把下面cwd的路径换成你的训练数据文件的路径就好了

cwd = "H:\\data\\train\\"

writer = tf.python_io.TFRecordWriter('train.tfrecords') #输出成tfrecord文件

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

for index in os.listdir(cwd):
    class_path=cwd+index+"\\"
    for img_name in os.listdir(class_path):
        img_path=class_path+img_name
        img = Image.open(img_path)
        img_raw = img.tobytes()
        example = tf.train.Example(features = tf.train.Features(feature = {
                                                                           "label": _int64_feature(int(index)),
                                                                           "img_raw": _bytes_feature(img_raw),
                                                                           }))
        writer.write(example.SerializeToString())
