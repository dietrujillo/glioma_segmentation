import glob
import os
import shutil
from typing import AnyStr, List, Tuple, Any, Callable, Dict

import dask
import nibabel as nib
import numpy as np
from dask.diagnostics import ProgressBar
from scipy.ndimage import zoom

from definitions import (
    DATA_PATH, PREPROCESSED_DATA_PATH, SCAN_TYPES,
    CROP_LIMIT, SEGMENTATION_CATEGORIES, SEGMENTATION_MERGE_DICT,
    CROP_SHAPE, RESIZE_SHAPE, RANDOM_SEED
)


def remove_background_fuzz(transformation: Callable, background_val: float = 0.0, tol: float = .1):
    """
    Remove background fuzz added by an interpolation transformation.
    Values in [background_val - tol, background_val + tol] will be reset to background_val.
    :param transformation: transformation function.
    :param background_val: value to be considered as background.
    :param tol: tolerance.
    :return:
    """
    def _remove_background_fuzz(arr: np.ndarray, **kwargs) -> np.ndarray:
        arr = transformation(arr, **kwargs)
        arr[(arr >= background_val - tol) & (arr <= background_val + tol)] = background_val
        return arr

    return _remove_background_fuzz


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
    return remove_background_fuzz(zoom)(arr, zoom=factor)


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
        out[..., index][arr == cat] = 1
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
            arr[..., index] |= arr[..., value]

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
def preprocess_patient(patient_dir: AnyStr, output_dir: AnyStr, process_seg: bool = True) -> None:
    """
    Preprocess a whole patient, applying both the MRI scans and the
    segmentation ground truth their respective pipelines.
    :param patient_dir: directory with patient files.
    :param output_dir: directory for the results of the pipeline.
    :param process_seg: whether to try to preprocess segmentation labels.
    :return: None
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        for scan_type in SCAN_TYPES:
            matches = glob.glob(os.path.join(patient_dir, f"*{scan_type}.nii.gz"))
            assert len(matches) == 1
            preprocess_scan(matches[0], os.path.join(output_dir,
                                                     os.path.basename(matches[0]).replace(".gz", "")))

        if process_seg:
            matches = glob.glob(os.path.join(patient_dir, f"*seg.nii.gz"))
            assert len(matches) == 1
            preprocess_segmentation(matches[0], os.path.join(output_dir,
                                                             os.path.basename(matches[0]).replace(".gz", "")))
    except AssertionError:
        print(f"\nPatient {os.path.basename(patient_dir)} had errors when loading some files.")


def preprocessing_pipeline(data_path: AnyStr = DATA_PATH,
                           output_path: AnyStr = PREPROCESSED_DATA_PATH,
                           process_seg: bool = True) -> None:
    """
    Parallelize the `preprocess_patient` function for every patient.
    :param data_path: input directory containing all patient dirs.
    :param output_path: output directory for the preprocessing results.
    :param process_seg: whether to try to preprocess segmentation labels.
    :return: None
    """
    delayed_operations = []
    for patient_dir in os.listdir(data_path):
        patient_input_path = os.path.join(data_path, patient_dir)
        if os.path.isdir(patient_input_path):
            patient_output_path = os.path.join(output_path, patient_dir)
            delayed_operations.append(preprocess_patient(patient_input_path,
                                                         patient_output_path,
                                                         process_seg=process_seg))

    with ProgressBar():
        dask.compute(delayed_operations)


def split_datasets(files: List[AnyStr],
                   output_path: AnyStr,
                   test_ratio: float = 0.2,
                   random_seed: int = RANDOM_SEED) -> None:
    """
    Split preprocessed datasets into training and test sets.
    :param files: paths pointing to preprocessed data.
    :param output_path: path to contain the training and test directories.
    :param test_ratio: proportion of the whole dataset to assign to test set. Must be a float between 0 and 1.
    :param random_seed: random seed.
    :return: None.
    """
    random_state = np.random.RandomState(random_seed)
    random_state.shuffle(files)

    limit = int(len(files) * test_ratio)

    os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "test"), exist_ok=True)

    datasets = {
        "train": files[limit:],
        "test": files[:limit]
    }

    for dataset_name, file_names in datasets.items():
        for dataset_file in file_names:
            shutil.move(dataset_file, os.path.join(output_path, dataset_name, os.path.basename(dataset_file)))


if __name__ == '__main__':

    data_dirs = [
        "MICCAI_BraTS2020_TrainingData/HGG",
        "MICCAI_BraTS2020_TrainingData/LGG",
    ]

    for data_dir in data_dirs:
        print(f"Preprocessing {data_dir}")
        preprocessing_pipeline(os.path.join(DATA_PATH, data_dir),
                               os.path.join(PREPROCESSED_DATA_PATH, data_dir))

    total_files = []

    for data_dir in data_dirs:
        for file in os.listdir(os.path.join(PREPROCESSED_DATA_PATH, data_dir)):
            total_files.append(os.path.join(PREPROCESSED_DATA_PATH, data_dir, file))

    # split_datasets moves every file to train and test directories
    split_datasets(files=total_files,
                   test_ratio=0.2,
                   output_path=PREPROCESSED_DATA_PATH)
