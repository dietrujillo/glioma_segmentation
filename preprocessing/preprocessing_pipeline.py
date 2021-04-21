import glob
import os
from typing import AnyStr, List, Tuple, Dict

import dask
import nibabel as nib
import numpy as np
from dask.diagnostics import ProgressBar

from definitions import (
    DATA_PATH, PREPROCESSED_DATA_PATH, SCAN_TYPES,
    CROP_LIMIT, SEGMENTATION_CATEGORIES, SEGMENTATION_MERGE_DICT
)


def crop(arr: np.array, lim: List[Tuple[int, int]] = CROP_LIMIT):
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
    min_value = np.min(arr)
    max_value = np.max(arr)
    return (arr - min_value) / (max_value - min_value)


def preprocess_scan(filepath: AnyStr, output_file: AnyStr) -> None:
    """
    Apply preprocessing pipeline to the MRI scan files.
    :param filepath: path to the MRI scan files.
    :param output_file: path in which to save pipeline output.
    :return: None
    """
    data = nib.load(filepath).get_fdata()
    data = crop(data)
    data = scale(data)
    nib.save(nib.Nifti1Image(data, None), output_file)


def one_hot_encode_segmentation(arr: np.ndarray, categories=None, handle_unknown="error") -> np.ndarray:
    """
    One-Hot (dummy) encode the segmentation volume.
    A volume of shape (x, y, z) with discrete values representing segmentation classes
    will be encoded as an (x, y, z, number_of_classes) hypercube with binary values (0, 1)
    such that arr[x, y, z, :] will be the class vector for the specified voxel.
    :param arr: array to be encoded.
    :param categories: expected categories. If None, extract categories from the data.
    :param handle_unknown: {"error", "ignore"}. How to respond to unexpected values if categories is specified.
    :return: encoded array.
    """
    if categories is None:
        categories = np.unique(arr)
    elif handle_unknown == "error":
        if set(np.unique(arr)) != set(categories):
            raise ValueError(f"Encountered unexpected value in array: values found are {np.unique(arr)}")

    out = np.zeros((*arr.shape, len(categories)), dtype=int)
    for index, cat in enumerate(categories):
        out[arr == cat][:, :, :, index] = 1
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
    seg = merge_segmentation_classes(seg)
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
