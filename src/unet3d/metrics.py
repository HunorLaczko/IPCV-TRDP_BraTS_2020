from functools import partial

from keras import backend as K
import tensorflow as tf
import numpy as np


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    y_true = K.cast(y_true,"float32")
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


def waveloss(ValInc=0.1, SpaInc=8, ValW, SpaW, NumSteps=10):
    def _waveloss(InputGT, Enhanced):  
        InputGT = K.cast(InputGT,"float32")

        IntersSection = tf.math.minimum(Enhanced, InputGT)
        Union = tf.math.maximum(Enhanced, InputGT)

        CurrentWave = tf.math.minimum(Enhanced, InputGT)
        WaveLoss = 0

        for step in range(NumSteps):
            Growed = CurrentWave + ValInc
            Growed = tf.math.minimum(Growed, Union)
            ValueDiff = tf.reduce_sum(Growed - CurrentWave)
            Growed = tf.nn.max_pool3d(Growed, SpaInc, 1, padding='SAME', data_format='NCDHW')
            Growed = tf.math.minimum(Growed, Union)
            TopologyDiff = tf.reduce_sum(Growed - CurrentWave)
            CurrentWave = Growed
            WaveLoss = WaveLoss + ValW[step] * ValueDiff + SpaW[step] * TopologyDiff

        return WaveLoss
    return _waveloss


def labelwise_waveloss(y_true, y_pred):
      waveloss(y_true[:,:,:,:, 0:1], y_pred[:,:,:,:, 0:1]) + waveloss(y_true[:,:,:,:, 1:2], y_pred[:,:,:,:, 1:2]) + waveloss(y_true[:,:,:,:, 2:], y_pred[:,:,:,:, 2:])


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
