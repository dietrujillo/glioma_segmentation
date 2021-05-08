import tensorflow as tf

from definitions import PREPROCESSED_DATA_SHAPE, LOSS_WEIGHTS

LOSS_WEIGHTS_TENSOR = tf.stack([tf.fill(PREPROCESSED_DATA_SHAPE, loss_weight)
                                for loss_weight in LOSS_WEIGHTS], axis=-1)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, epsilon: float = 1e-10) -> tf.Tensor:
    """
    Dice loss implementation. See [arXiv:2010.13499v1] for details.
    Input predictions and ground truth arrays are expected to be binary 0-1 values.
    Thus, when calculating the dot product of one array with itself, we simply sum the values.
    :param y_true: ground truth array.
    :param y_pred: predictions array.
    :param epsilon: smoothness factor.
    :return: differentiable dice loss.
    """
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    epsilon = tf.constant(epsilon)
    numerator = tf.add(intersection, epsilon)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + epsilon
    general_loss = tf.constant(1.) - tf.constant(2.) * tf.divide(numerator, denominator)
    per_pixel_loss = general_loss * tf.square(tf.subtract(y_true, y_pred))
    loss_weights = tf.stack([LOSS_WEIGHTS_TENSOR] * y_true.shape[0], axis=0)
    return tf.multiply(per_pixel_loss, loss_weights)
