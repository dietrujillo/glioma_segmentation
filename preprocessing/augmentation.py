from typing import Callable, Tuple

import numpy as np
from scipy.ndimage import rotate
from definitions import SCAN_TYPES, NO_AUGMENTATION_PROBABILITY


def _augmentation(apply_to_label: bool = False) \
        -> Callable[[Callable[[np.ndarray], np.ndarray]],
                    Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """
    Decorator used to apply an augmentation to many elements at once.

    Functions using this decorator should have one array as input. However, once the decorator has been
    applied, the inputs should be a list of arrays for every resonance type, and an array containing
    the ground truth segmentation.

    :param apply_to_label: whether to apply the augmentation to the ground truth segmentation.
    :return: a decorated function with the characteristics described above.
    """

    def augmentation_inner(augment_function: Callable[[np.ndarray], np.ndarray]) \
            -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

        def augment_many(data: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            for index in range(len(SCAN_TYPES)):
                data[index] = augment_function(data[index])
            if apply_to_label:
                seg = augment_function(seg)
            return data, seg
        return augment_many

    return augmentation_inner


@_augmentation(apply_to_label=True)
def _sagittal_flip(arr: np.ndarray) -> np.ndarray:
    """
    Flip along the first axis. For 3D brain MRI data, this means the sagittal plane.
    :param arr: array to be flipped.
    :return: array flipped on the sagittal plane.
    """
    return arr[::-1]


@_augmentation(apply_to_label=False)
def _gaussian_noise(arr: np.ndarray) -> np.ndarray:
    """
    Apply gaussian noise to an array.

    All calculations are done excluding zero values,
    as to avoid taking the background of the image into account.

    Values of noise resulting in array values outside of the nominal range
    (i.e. [Q1 - 3/2 * IQR, Q3 + 3/2 * IQR],
    where Q1 and Q3 are the first and third quartiles respectively and IQR is the interquartile range)
    are excluded and not applied.

    :param arr: array to be noised.
    :return: copy of the array, modified with gaussian noise.
    """
    q1 = np.percentile(arr[arr != 0], 25)
    q3 = np.percentile(arr[arr != 0], 75)
    iqr = q3 - q1
    nominal_range = q1 - (1.5 * iqr), q3 + (1.5 * iqr)

    sigma = arr[arr != 0].std()
    noise = np.where(arr == 0, 0, np.random.normal(0, sigma, size=arr.shape))

    ret = np.where((nominal_range[0] <= arr + noise) & (arr + noise <= nominal_range[1]), arr + noise, arr)
    return ret


@_augmentation(apply_to_label=True)
def _rotation(arr: np.ndarray, min_degree=-1, max_degree=1) -> np.ndarray:
    """
    Rotate the array along all axes in a range of (min_degree, max_degree)
    :param arr: array to be modified.
    :param min_degree: lower bound for rotation in degrees.
    :param max_degree: upper bound for rotation in degrees.
    :return: modified array.
    """
    # FIXME: a different random rotation is applied to each scan for the same patient.
    #  They should share the rotation degrees.
    axes = list(range(arr.ndim))
    rotating_axis = np.random.choice(axes)
    rotating_plane = axes[:rotating_axis] + axes[rotating_axis + 1:]
    degree = np.random.uniform(min_degree, max_degree)
    return rotate(arr, degree, rotating_plane)


_AUGMENTATIONS = (_sagittal_flip, _gaussian_noise, _rotation)


def apply_augmentation(data: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly apply an augmentation to the data, or no augmentation at all.
    The probability distribution is NO_AUGMENTATION_PROBABILITY for no transformation,
    then uniformly distributed across the possible transformations.

    :param data: list of the different MRI scans.
    :param seg: ground truth segmentation.
    :return: tuple with data and seg, transformed or otherwise.
    """
    if np.random.rand() <= NO_AUGMENTATION_PROBABILITY:
        return data, seg
    else:
        return np.random.choice(_AUGMENTATIONS)(data, seg)