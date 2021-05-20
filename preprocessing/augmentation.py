from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.stats import percentile
from scipy.ndimage import rotate

from definitions import SCAN_TYPES, ROTATION_MAX_DEGREES
from preprocessing.preprocessing_pipeline import remove_background_fuzz


def augmentation(apply_to_label: bool = False) \
        -> Callable[[Callable[[tf.Tensor, np.random.RandomState], tf.Tensor]],
                    Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]]:
    """
    Decorator used to apply an augmentation to many elements at once.

    Functions using this decorator should have one array as input. However, once the decorator has been
    applied, the inputs should be a list of arrays for every resonance type, and an array containing
    the ground truth segmentation.

    A random integer will be provided as a RandomState initializer for the augmentations,
    so that every image from the same patient uses the same random numbers.

    :param apply_to_label: whether to apply the augmentation to the ground truth segmentation.
    :return: a decorated function with the characteristics described above.
    """

    def augmentation_inner(augment_function: Callable[..., tf.Tensor]) \
            -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:

        def augment_many(data: tf.Tensor, seg: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            data = tf.identity(data)
            random_state = np.random.randint(65535)

            augmented_data = []
            for index in range(len(SCAN_TYPES)):
                augmented_data.append(augment_function(data[..., index], random_state=random_state))
            data = tf.stack(augmented_data, axis=-1)
            if apply_to_label:
                seg = augment_function(seg, random_state=random_state)
                seg = tf.clip_by_value(tf.round(seg), 0, 1)
            return data, seg
        return augment_many

    return augmentation_inner


@augmentation(apply_to_label=True)
@tf.function
def sagittal_flip(tensor: tf.Tensor, random_state: int = None) -> tf.Tensor:
    """
    Flip along the first axis. For 3D brain MRI data, this means the sagittal plane.
    :param tensor: tensor to be flipped.
    :param random_state: random number generator. Unused, kept for compatibility.
    :return: tensor flipped on the sagittal plane.
    """
    return tf.reverse(tensor, axis=tf.constant([0]))


@augmentation(apply_to_label=False)
@tf.function
def gaussian_noise(tensor: tf.Tensor, random_state: int) -> tf.Tensor:
    """
    Apply gaussian noise to an array.

    All calculations are done excluding zero values,
    as to avoid taking the background of the image into account.

    Values of noise resulting in array values outside of the nominal range
    (defined as [Q1 + 0.1 * IQR, Q3 - 0.1 * IQR] for this problem,
    where Q1 and Q3 are the first and third quartiles respectively and IQR is the interquartile range)
    are excluded and not applied.

    :param tensor: tensor to be noised.
    :param random_state: random seed.
    :return: copy of the tensor, modified with gaussian noise.
    """

    q1 = percentile(tensor[tensor != 0], 25)
    q3 = percentile(tensor[tensor != 0], 75)
    iqr = q3 - q1
    nominal_range = q1 + (0.1 * iqr), q3 - (0.1 * iqr)

    sigma = 0.1
    noise = tf.where(tensor == 0, tf.zeros_like(tensor, dtype=tensor.dtype),
                     tf.random.normal(shape=tensor.shape, mean=0, stddev=sigma, seed=random_state, dtype=tensor.dtype))

    noised_tensor = tensor + noise
    ret = tf.where((nominal_range[0] <= noised_tensor) & (noised_tensor <= nominal_range[1]), noised_tensor, tensor)
    return ret


@augmentation(apply_to_label=True)
def rotation(tensor: tf.Tensor, random_state: int) -> tf.Tensor:
    """
    Rotate the tensor along all axes in a range of (min_degree, max_degree)
    :param tensor: tensor to be modified.
    :param random_state: random state.
    :return: modified array.
    """
    random_state = np.random.RandomState(random_state)

    degree = random_state.uniform(*ROTATION_MAX_DEGREES)
    rotation_plane = tuple(random_state.choice(3, replace=False, size=2))
    rotated_tensor = remove_background_fuzz(rotate)(tensor.numpy(), angle=degree, axes=rotation_plane, reshape=False)
    return tf.cast(np.clip(rotated_tensor, 0, 1), dtype=tensor.dtype)


def compose(*ops: Callable) -> Callable:
    """
    Apply a number of operations sequentially.
    :param ops: operations to apply
    :return: callable with composition of the operations.
    """
    def composed_transform(*inputs: tf.Tensor) -> Tuple[tf.Tensor]:
        outputs = inputs
        for op in ops:
            outputs = op(*outputs)
        return outputs

    return composed_transform


def one_of(*ops: Callable, prob: np.ndarray = None) -> Callable:
    """
    Randomly apply one of a number of operations.
    :param ops: possible operations to apply.
    :param prob: probability weights for each operation.
    :return: callable with one of the operations chosen randomly.
    """
    def transform(*inputs: tf.Tensor) -> Tuple[tf.Tensor]:
        op = np.random.choice(ops, p=prob)
        return op(*inputs)
    return transform


def optional(op: Callable, prob: float = 0.5) -> Callable:
    """
    Apply an operation with probability prob.
    :param op: operation to apply.
    :param prob: probability of applying the operation.
    :return: callable that randomly applies op or does nothing.
    """
    def optional_transform(*inputs: tf.Tensor) -> Tuple[tf.Tensor]:
        if np.random.rand() < prob:
            return op(*inputs)
        return inputs

    return optional_transform


AUGMENTATION_PIPELINE = compose(
    optional(sagittal_flip, prob=0.5),
    optional(
        one_of(
            rotation,
            gaussian_noise
        ),
        prob=0.5
    )
)


def apply_augmentation(data: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the augmentation pipeline to some data.

    :param data: list of the different MRI scans.
    :param seg: ground truth segmentation.
    :return: tuple with data and seg, after being randomly modified.
    """
    return AUGMENTATION_PIPELINE(data, seg)
