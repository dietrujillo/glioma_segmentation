import tensorflow as tf


def dice_score(y_true, y_pred, epsilon=1e-10):
    """
    Dice score coefficient.
    :param y_true: ground truth.
    :param y_pred: predictions.
    :param epsilon: smoothness factor.
    :return: dice score coefficient.
    """
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    epsilon = tf.constant(epsilon)
    numerator = tf.add(intersection, tf.constant(epsilon))
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + epsilon
    return tf.constant(2) * tf.divide(numerator, denominator)


def dice_whole_tumor(y_true, y_pred):
    """
    Compute dice score for the whole tumor.
    :param y_true: ground truth.
    :param y_pred: predictions.
    :return: Dice Score for the whole tumor.
    """
    return dice_score(y_true[:, :, :, 1], y_pred[:, :, :, 1])


def dice_tumor_core(y_true, y_pred):
    """
    Compute dice score for the tumor core.
    :param y_true: ground truth.
    :param y_pred: predictions.
    :return: Dice Score for the tumor core.
    """
    return dice_score(y_true[:, :, :, 2], y_pred[:, :, :, 2])


def dice_enhancing_tumor_core(y_true, y_pred):
    """
    Compute dice score for the enhancing tumor core.
    :param y_true: ground truth.
    :param y_pred: predictions.
    :return: Dice Score for the enhancing tumor core.
    """
    return dice_score(y_true[:, :, :, 0], y_pred[:, :, :, 0])


METRICS = [dice_whole_tumor, dice_tumor_core, dice_enhancing_tumor_core]