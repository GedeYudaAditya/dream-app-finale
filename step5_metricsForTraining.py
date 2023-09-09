"""
Code By Gus Yudha
Tanggal 1 Mei 2022

Perlengkapan
1. Package numpy, tensorflow 2.8.0, dan Keras 2.4.3
2. Bekerja di OS Windows
3. Pyhton 3.8

Keterangan
1. Tempatkan pada folder yang sama dengan code trainByGusYudha dan code evaluationByGusYudha
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def iou(yBenar, yTebak):
    def f(yBenar, yTebak):
        irisan = (yBenar * yTebak).sum()
        lapisan = yBenar.sum() + yTebak.sum() - irisan
        x = (irisan + 1e-15) / (lapisan + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [yBenar, yTebak], tf.float32)

def iou_evaluation(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (sum(y_true_f) + sum(y_pred_f) - intersection + 1.0)

def iou_loss(y_true, y_pred):
    iou_result = iou_evaluation(y_true,y_pred)
    return 1-iou_result

smooth = 1e-15
def dice_coef(yBenar, yTebak):
    yBenar = tf.keras.layers.Flatten()(yBenar)
    yTebak = tf.keras.layers.Flatten()(yTebak)
    irisan = tf.reduce_sum(yBenar * yTebak)
    return (2. * irisan + smooth) / (tf.reduce_sum(yBenar) + tf.reduce_sum(yTebak) + smooth)

def dice_loss(yBenar, yTebak):
    return 1.0 - dice_coef(yBenar, yTebak)
