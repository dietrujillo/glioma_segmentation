import tensorflow as tf
from definitions import LOSS_WEIGHTS


def dice_score(y_true, y_pred, epsilon=1e-10):
    """
    Dice score coefficient.
    :param y_true: ground truth.
    :param y_pred: predictions.
    :param epsilon: smoothness factor.
    :return: dice score coefficient.
    """
    y_pred = tf.round(y_pred)
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    epsilon = tf.constant(epsilon)
    numerator = tf.add(intersection, epsilon)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + epsilon
    return tf.constant(2.) * tf.divide(numerator, denominator)


def dice_wt(y_true, y_pred):
    """
    Compute dice score for the whole tumor.
    :param y_true: ground truth.
    :param y_pred: predictions.
    :return: Dice Score for the whole tumor.
    """
    return dice_score(y_true[:, :, :, :, 1], y_pred[:, :, :, :, 1])


def dice_tc(y_true, y_pred):
    """
    Compute dice score for the tumor core.
    :param y_true: ground truth.
    :param y_pred: predictions.
    :return: Dice Score for the tumor core.
    """
    return dice_score(y_true[:, :, :, :, 2], y_pred[:, :, :, :, 2])


def dice_etc(y_true, y_pred):
    """
    Compute dice score for the enhancing tumor core.
    :param y_true: ground truth.
    :param y_pred: predictions.
    :return: Dice Score for the enhancing tumor core.
    """
    return dice_score(y_true[:, :, :, :, 0], y_pred[:, :, :, :, 0])


def weighted_dice_score(y_true, y_pred):
    scores = [dice_wt(y_true, y_pred), dice_tc(y_true, y_pred), dice_etc(y_true, y_pred)]
    return sum(LOSS_WEIGHTS[i] * score for i, score in enumerate(scores))


METRICS = [weighted_dice_score, dice_wt, dice_tc, dice_etc]
