import glob
import os
from typing import AnyStr, List, Tuple, Any, Callable, Dict

import dask
import nibabel as nib
import numpy as np
from dask.diagnostics import ProgressBar

from scipy.ndimage import zoom

from definitions import (
    DATA_PATH, PREPROCESSED_DATA_PATH, SCAN_TYPES,
    CROP_LIMIT, SEGMENTATION_CATEGORIES, SEGMENTATION_MERGE_DICT,
    CROP_SHAPE, RESIZE_SHAPE
)


def remove_interpolated_background(transformation: Callable, background_val: float = 0.0, tol: float = .1):

    def _remove_background(arr: np.ndarray, **kwargs) -> np.ndarray:
        arr = transformation(arr, **kwargs)
        arr[(arr >= background_val - tol) & (arr <= background_val + tol)] = background_val
        return arr

    return _remove_background


def resize(arr: np.ndarray,
           input_shape: Tuple[int, int, int] = CROP_SHAPE,
           output_shape: Tuple[int, int, int] = RESIZE_SHAPE) -> np.ndarray:
    """
    Resize a 3D array.
    :param arr: array to be resized.
    :param input_shape: expected shape of the received array.
    :param output_shape: shape of the resized array.
    :return: resized array.
    """
    factor = [b / a for a, b in zip(input_shape, output_shape)]
    return remove_interpolated_background(zoom)(arr, zoom=factor)


def crop(arr: np.array, lim: List[Tuple[int, int]] = CROP_LIMIT) -> np.ndarray:
    """
    Crop a 3D array using the limits provided in CROP_LIMIT
    :param arr: array to be cropped
    :param lim: list containing tuples of lower and upper limit per axis.
    :return: cropped array.
    """
    return arr[lim[0][0]:lim[0][1], lim[1][0]:lim[1][1], lim[2][0]:lim[2][1]]


def scale(arr: np.ndarray) -> np.ndarray:
    """
    Simple [0, 1] scaling.
    :param arr: array to be scaled.
    :return: scaled array.
    """
    min_value = arr[0, 0, 0]
    max_value = np.max(arr)
    return np.clip((arr - min_value) / (max_value - min_value), 0, 1)


def preprocess_scan(filepath: AnyStr, output_file: AnyStr) -> None:
    """
    Apply preprocessing pipeline to the MRI scan files.
    :param filepath: path to the MRI scan files.
    :param output_file: path in which to save pipeline output.
    :return: None
    """
    data = nib.load(filepath).get_fdata()
    data = crop(data)
    data = resize(data)
    data = scale(data)
    data = data.astype("float32")
    nib.save(nib.Nifti1Image(data, None), output_file)


def one_hot_encode_segmentation(arr: np.ndarray, categories: List[Any] = SEGMENTATION_CATEGORIES,
                                handle_unknown: AnyStr = "error",
                                use_background: bool = False, background_value: float = 0) -> np.ndarray:
    """
    One-Hot (dummy) encode the segmentation volume.
    A volume of shape (x, y, z) with discrete values representing segmentation classes
    will be encoded as an (x, y, z, number_of_classes) hypercube with binary values (0, 1)
    such that arr[x, y, z, :] will be the class vector for the specified voxel.
    :param arr: array to be encoded.
    :param categories: expected categories. If None, extract categories from the data.
    :param handle_unknown: {"error", "ignore"}. How to respond to unexpected values if categories is specified.
    :param use_background: whether to use the background of the image as another category.
    :param background_value: value for the background voxels.
    :return: encoded array.
    """
    if categories is None:
        categories = list(np.unique(arr))
    elif handle_unknown == "error":
        if any(x not in categories for x in np.unique(arr)):
            raise ValueError(f"Encountered unexpected value in array: values found are {np.unique(arr)}")

    if not use_background:
        categories = list(categories)
        categories.remove(background_value)

    out = np.zeros((*arr.shape, len(categories)), dtype=int)
    for index, cat in enumerate(categories):
        out[:, :, :, index][arr == cat] = 1
    return out


def merge_segmentation_classes(arr: np.ndarray,
                               merge_dict: Dict[int, Tuple[int]] = SEGMENTATION_MERGE_DICT) -> np.ndarray:
    """
    Merges some segmentation classes together.
    For example, classes 1, 2 and 4 combine to form the whole tumor.
    :param arr: array to be modified.
    :param merge_dict: segmentation classes to merge.
    :return: modified array.
    """
    for index, values in merge_dict.items():
        for value in values:
            arr[:, :, :, index] |= arr[:, :, :, value]

    return arr


def preprocess_segmentation(filepath: AnyStr, output_file: AnyStr) -> None:
    """
    Apply preprocessing pipeline to the segmentation ground truth.
    :param filepath: path to the segmentation ground truth.
    :param output_file: path in which to save pipeline output.
    :return: None
    """
    seg = nib.load(filepath).get_fdata()
    seg = crop(seg)
    seg = one_hot_encode_segmentation(seg, categories=SEGMENTATION_CATEGORIES)
    seg = np.stack([np.around(resize(seg[..., i])) for i in range(seg.shape[-1])], axis=-1)
    seg = np.clip(seg, 0, 1)
    seg = merge_segmentation_classes(seg)
    seg = seg.astype(int)
    nib.save(nib.Nifti1Image(seg, None), output_file)


@dask.delayed
def preprocess_patient(patient_dir: AnyStr, output_dir: AnyStr) -> None:
    """
    Preprocess a whole patient, applying both the MRI scans and the
    segmentation ground truth their respective pipelines.
    :param patient_dir: directory with patient files.
    :param output_dir: directory for the results of the pipeline.
    :return: None
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        for scan_type in SCAN_TYPES:
            matches = glob.glob(os.path.join(patient_dir, f"*{scan_type}.nii.gz"))
            assert len(matches) == 1
            preprocess_scan(matches[0], os.path.join(output_dir, os.path.basename(matches[0])))

        matches = glob.glob(os.path.join(patient_dir, f"*seg.nii.gz"))
        assert len(matches) == 1
        preprocess_segmentation(matches[0], os.path.join(output_dir, os.path.basename(matches[0])))
    except AssertionError:
        print(f"\nPatient {os.path.basename(patient_dir)} had errors when loading some files.")


def preprocessing_pipeline(data_path: AnyStr = DATA_PATH,
                           output_path: AnyStr = PREPROCESSED_DATA_PATH) -> None:
    """
    Parallelize the `preprocess_patient` function for every patient.
    :param data_path: input directory containing all patient dirs.
    :param output_path: output directory for the preprocessing results.
    :return: None
    """
    delayed_operations = []
    for patient_dir in os.listdir(data_path):
        patient_input_path = os.path.join(data_path, patient_dir)
        if os.path.isdir(patient_input_path):
            patient_output_path = os.path.join(output_path, patient_dir)
            delayed_operations.append(preprocess_patient(patient_input_path, patient_output_path))

    with ProgressBar():
        dask.compute(delayed_operations)


if __name__ == '__main__':
    preprocessing_pipeline(os.path.join(DATA_PATH, "MICCAI_BraTS2020_TrainingData/HGG"),
                           os.path.join(PREPROCESSED_DATA_PATH, "MICCAI_BraTS2020_TrainingData/HGG"))

    preprocessing_pipeline(os.path.join(DATA_PATH, "MICCAI_BraTS2020_TrainingData/LGG"),
                           os.path.join(PREPROCESSED_DATA_PATH, "MICCAI_BraTS2020_TrainingData/LGG"))

    preprocessing_pipeline(os.path.join(DATA_PATH, "MICCAI_BraTS_2019_Data_Training/HGG"),
                           os.path.join(PREPROCESSED_DATA_PATH, "MICCAI_BraTS_2019_Data_Training/HGG"))

    preprocessing_pipeline(os.path.join(DATA_PATH, "MICCAI_BraTS_2019_Data_Training/LGG"),
                           os.path.join(PREPROCESSED_DATA_PATH, "MICCAI_BraTS_2019_Data_Training/LGG"))

    preprocessing_pipeline(os.path.join(DATA_PATH, "MICCAI_BraTS_2018_Data_Training/HGG"),
                           os.path.join(PREPROCESSED_DATA_PATH, "MICCAI_BraTS_2018_Data_Training/HGG"))

    preprocessing_pipeline(os.path.join(DATA_PATH, "MICCAI_BraTS_2018_Data_Training/LGG"),
                           os.path.join(PREPROCESSED_DATA_PATH, "MICCAI_BraTS_2018_Data_Training/LGG"))
