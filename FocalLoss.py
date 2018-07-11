import keras.backend as K
import numpy as np


def Focal_Loss(y_true, y_pred, alpha=0.25, gamma=2):
    """
    focal loss for multi-class classification
    fl(pt) = -alpha*(1-pt)^(gamma)*log(pt)

    :param y_true: ground truth one-hot vector shape of [batch_size, nb_class]
    :param y_pred: prediction after softmax shape of [batch_size, nb_class]
    :param alpha:
    :param gamma:
    :return:
    """
    # # parameters
    # alpha = 0.25
    # gamma = 2

    # To avoid divided by zero
    y_pred += K.epsilon()

    # Cross entropy
    ce = -y_true * np.log(y_pred)

    # Not necessary to multiply y_true(cause it will multiply with CE which has set unconcerned index to zero ),
    # but refer to the definition of p_t, we do it
    weight = np.power(1 - y_pred, gamma) * y_true

    # Now fl has a shape of [batch_size, nb_class]
    # alpha should be a step function as paper mentioned, but it doesn't matter like reason mentioned above
    # (CE has set unconcerned index to zero)
    #
    # alpha_step = tf.where(y_true, alpha*np.ones_like(y_true), 1-alpha*np.ones_like(y_true))
    fl = ce * weight * alpha

    # Both reduce_sum and reduce_max are ok
    reduce_fl = K.max(fl, axis=-1)

    return reduce_fl


