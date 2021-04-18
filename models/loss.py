from typing import Dict

import numpy as np
import tensorflow as tf

from definitions import LOSS_WEIGHTS


def dice_loss(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
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
    numerator = tf.add(intersection, tf.constant(epsilon))
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + epsilon
    return tf.constant(1) - tf.constant(2) * tf.divide(numerator, denominator)


def generalized_dice_loss(y_true: np.ndarray, y_pred: np.ndarray,
                          segmentation_classes: int = 4,
                          loss_weights: Dict[int, float] = LOSS_WEIGHTS) -> float:
    """
    Generalized dice loss for multiple classes.
    Every class predictions and ground truth are converted to binary arrays.
    Then, the dice loss for every array is added to the total loss (with a weight).
    :param y_true: ground truth array.
    :param y_pred: predictions array.
    :param segmentation_classes: number of expected different classes.
    :param loss_weights: weights for every class. All of its values must add up to 1.
    :return: generalized dice loss for multiple classes.
    """
    loss = 0
    for segmentation_class in range(segmentation_classes):
        class_true = tf.where(y_true == segmentation_class, 1, 0)
        class_predictions = tf.where(y_pred == segmentation_class, 1, 0)
        loss += loss_weights[segmentation_class] * dice_loss(class_true, class_predictions)
    return loss
