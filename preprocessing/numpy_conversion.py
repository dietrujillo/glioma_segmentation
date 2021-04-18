import glob
from typing import *

import dask
import nibabel as nib
import numpy as np
from dask.diagnostics import ProgressBar

from definitions import *


def nii_to_npy(patient_dir: AnyStr, patient_index: int, output_path: AnyStr) -> None:
    """
    Converts many .nii files into one NumPy file per patient.
    :param patient_dir: path to patient MRI data.
    :param patient_index: patient identifier.
    :param output_path: where to save the resulting .npy files.
    :return: None
    """
    patient_array = np.zeros((len(SCAN_TYPES) + 1, *INPUT_DATA_SHAPE))
    try:
        for index, scan_type in enumerate(SCAN_TYPES):
            matches = glob.glob(os.path.join(patient_dir, f"*{scan_type}.nii"))
            assert len(matches) == 1
            patient_array[index] = nib.load(matches[0]).get_fdata()

        matches = glob.glob(os.path.join(patient_dir, f"*seg.nii"))
        assert len(matches) == 1
        patient_array[-1] = nib.load(matches[0]).get_fdata()
    except AssertionError:
        print(f"\nPatient {os.path.basename(patient_dir)} had errors when loading some files.")
    else:
        np.save(os.path.join(output_path, f"patient{patient_index}.npy"), patient_array)


def process_numpy_conversion(input_data_path: AnyStr, output_data_path: AnyStr) -> None:
    """
    Uses dask parallelization to mass process all .nii files into NumPy arrays.
    :param input_data_path: location with directories containing .nii files.
    :param output_data_path: desired location for output .npy files.
    :return: None
    """
    os.makedirs(output_data_path, exist_ok=True)

    operations = []
    for patient_index, patient in enumerate(
            filter(lambda file: os.path.isdir(os.path.join(input_data_path, file)),
                   sorted(os.listdir(input_data_path)))):
        operations.append(dask.delayed(nii_to_npy)(os.path.join(input_data_path, patient),
                                                   patient_index, output_data_path))
    with ProgressBar():
        dask.compute(operations)


if __name__ == '__main__':
    process_numpy_conversion(input_data_path=ORIGINAL_DATA_PATH,
                             output_data_path=os.path.join(PROCESSED_DATA_PATH, "train"))