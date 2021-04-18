import os
from typing import Optional, AnyStr, Iterable, Generator, List, Tuple

import nibabel as nib
import numpy as np

from definitions import BATCH_SIZE, SCAN_TYPES
from preprocessing.augmentation import apply_augmentation


def data_loader(data_path: AnyStr,
                patients: Optional[Iterable[AnyStr]] = None,
                augment: bool = False,
                batch_size: int = BATCH_SIZE,
                shuffle_all: bool = True,
                shuffle_batch: bool = True,
                verbose: bool = False) \
        -> Generator[List[Tuple[np.ndarray, np.ndarray]], None, None]:
    """
    Loads data iteratively from disk.
    :param data_path: path to the data files.
    :param patients: which patient files to load.
    :param augment: whether to perform online data augmentation.
    :param batch_size: how many patients per batch to load.
    :param shuffle_all: whether to shuffle patients before loading anything
    :param shuffle_batch: whether to shuffle contents of batch before yielding (for online data augmentation)
    :param verbose: whether to inform of any missing files.
    :return: Generator yielding batches of tuples of type (np.ndarray, np.ndarray)
    """
    if patients is None:
        patients = sorted(os.listdir(data_path))

    if shuffle_all:
        np.random.shuffle(patients)

    batch = []
    current_batch = 0

    for patient in patients:
        patient_dir = os.path.join(data_path, patient)
        try:
            data = []
            for scan_type in SCAN_TYPES:
                scan = nib.load(os.path.join(patient_dir, f"{patient}_{scan_type}.nii.gz")).get_fdata()
                data.append(scan)

            data = np.stack(data, axis=-1)
            seg = nib.load(os.path.join(patient_dir, f"{patient}_seg.nii.gz")).get_fdata()

            if augment:
                data, seg = apply_augmentation(data, seg)

            batch.append((data, seg))
            if len(batch) >= batch_size:
                if shuffle_batch:
                    np.random.shuffle(batch)
                yield batch
                current_batch += 1
                batch = []

        except FileNotFoundError as e:
            if verbose:
                print(f"Missing file for patient {patient}: {e.filename}")

    if batch:
        if shuffle_batch:
            np.random.shuffle(batch)
        yield batch
