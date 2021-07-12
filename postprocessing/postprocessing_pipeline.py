from typing import List, Tuple

import dask
import numpy as np
from dask.diagnostics import ProgressBar
from scipy.ndimage import zoom

from definitions import (
    CROP_LIMIT, SEGMENTATION_CATEGORIES,
    CROP_SHAPE, RESIZE_SHAPE, INPUT_DATA_SHAPE
)
from preprocessing.preprocessing_pipeline import remove_background_fuzz


def undo_resize(arr: np.ndarray,
                input_shape: Tuple[int, int, int] = RESIZE_SHAPE,
                output_shape: Tuple[int, int, int] = CROP_SHAPE) -> np.ndarray:
    """
    Resize a 3D segmentation labels array to the original size after cropping.
    :param arr: array to be resized.
    :param input_shape: expected shape of the received array.
    :param output_shape: shape of the resized array.
    :return: resized array.
    """
    factor = [b / a for a, b in zip(input_shape, output_shape)]
    return remove_background_fuzz(zoom)(arr, zoom=factor)


def undo_crop(arr: np.array,
              shape: Tuple[int, int, int] = INPUT_DATA_SHAPE,
              lim: List[Tuple] = CROP_LIMIT) -> np.ndarray:
    """
    Undo 3D segmentation label cropping by zero-padding the limits.
    :param arr: array to be padded.
    :param shape: desired output shape.
    :param lim: limits used for cropping.
    :return: array padded with zeros in crop_limit
    """
    return np.pad(arr,
                  [(lim[0][0], shape[0] - lim[0][1]),
                   (lim[1][0], shape[1] - lim[1][1]),
                   (lim[2][0], shape[2] - lim[2][1])],
                  mode="constant",
                  constant_values=0)


def postprocess_segmentation(seg: np.ndarray) -> np.ndarray:
    """
    Apply postprocessing pipeline to the segmentation predictions.
    :param seg: predicted labels.
    :return: postprocessed labels.
    """
    
    seg = np.clip(np.around(seg), 0, 1)
    out = np.zeros(INPUT_DATA_SHAPE)

    for segmentation_class in [1, 2, 0]:
        before_resize = undo_resize(seg[..., segmentation_class])
        before_resize = np.clip(np.around(before_resize), 0, 1)
        before_crop = undo_crop(before_resize)
        out[before_crop == 1] = SEGMENTATION_CATEGORIES[segmentation_class + 1]

    out = np.around(out).astype(np.int32)

    return out
