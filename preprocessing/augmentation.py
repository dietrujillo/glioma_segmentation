from typing import Callable, Tuple

import numpy as np
from scipy.ndimage import rotate
from definitions import SCAN_TYPES, NO_AUGMENTATION_PROBABILITY, ROTATION_MAX_DEGREES


def _augmentation(apply_to_label: bool = False) \
        -> Callable[[Callable[[np.ndarray, np.random.RandomState], np.ndarray]],
                    Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
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

    def augmentation_inner(augment_function: Callable[..., np.ndarray]) \
            -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

        def augment_many(data: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            data = np.copy(data)
            random_state = np.random.randint(2 ** 16 - 1)
            for index in range(len(SCAN_TYPES)):
                data[..., index] = augment_function(data[..., index], random_state=random_state)
            if apply_to_label:
                seg = augment_function(seg, random_state=random_state)
                seg = np.clip(np.around(seg), 0, 1)
            return data, seg
        return augment_many

    return augmentation_inner


@_augmentation(apply_to_label=True)
def _sagittal_flip(arr: np.ndarray, random_state: int = None) -> np.ndarray:
    """
    Flip along the first axis. For 3D brain MRI data, this means the sagittal plane.
    :param arr: array to be flipped.
    :param random_state: random number generator. Unused, kept for compatibility.
    :return: array flipped on the sagittal plane.
    """
    return arr[::-1]


@_augmentation(apply_to_label=False)
def _gaussian_noise(arr: np.ndarray, random_state: int = None) -> np.ndarray:
    """
    Apply gaussian noise to an array.

    All calculations are done excluding zero values,
    as to avoid taking the background of the image into account.

    Values of noise resulting in array values outside of the nominal range
    (i.e. [Q1 - 3/2 * IQR, Q3 + 3/2 * IQR],
    where Q1 and Q3 are the first and third quartiles respectively and IQR is the interquartile range)
    are excluded and not applied.

    :param arr: array to be noised.
    :param random_state: random number generator. Unused, kept for compatibility.
    :return: copy of the array, modified with gaussian noise.
    """
    random_state = np.random.RandomState(random_state)

    q1 = np.percentile(arr[arr != 0], 25)
    q3 = np.percentile(arr[arr != 0], 75)
    iqr = q3 - q1
    nominal_range = q1 - (1.5 * iqr), q3 + (1.5 * iqr)

    sigma = arr[arr != 0].std()
    noise = np.where(arr == 0, 0, random_state.normal(0, sigma, size=arr.shape))

    ret = np.where((nominal_range[0] <= arr + noise) & (arr + noise <= nominal_range[1]), arr + noise, arr)
    return ret


@_augmentation(apply_to_label=True)
def _rotation(arr: np.ndarray, random_state: int = None) -> np.ndarray:
    """
    Rotate the array along all axes in a range of (min_degree, max_degree)
    :param arr: array to be modified.
    :param random_state: random state.
    :return: modified array.
    """
    random_state = np.random.RandomState(random_state)

    degree = random_state.uniform(*ROTATION_MAX_DEGREES)
    rotation_plane = tuple(random_state.choice(3, replace=False, size=2))
    return np.clip(rotate(arr, angle=degree, axes=rotation_plane, reshape=False), 0, 1)


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
