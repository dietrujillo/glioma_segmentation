import os
from typing import Optional, AnyStr, Iterable, Tuple, Union

import nibabel as nib
import numpy as np
import tensorflow as tf

from definitions import BATCH_SIZE, SCAN_TYPES
from preprocessing.augmentation import apply_augmentation


class BraTSDataLoader(tf.keras.utils.Sequence):
    """
    Data generator class for BraTS datasets.
    """
    def __init__(self,
                 data_path: AnyStr,
                 patients: Optional[Iterable[AnyStr]] = None,
                 augment: bool = False,
                 batch_size: int = BATCH_SIZE,
                 shuffle_all: bool = True,
                 shuffle_batch: bool = True,
                 retrieve_seg: bool = True,
                 compressed_files: bool = False,
                 verbose: bool = False):
        """
        Loads data iteratively from disk.
        :param data_path: path to the data files.
        :param patients: which patient files to load.
        :param augment: whether to perform online data augmentation.
        :param batch_size: how many patients per batch to load.
        :param shuffle_all: whether to shuffle patients before loading anything
        :param shuffle_batch: whether to shuffle patient order inside the batch.
        :param retrieve_seg: whether to try to retrieve segmentation labels.
        :param compressed_files: whether to look for .nii or .nii.gz files.
        :param verbose: whether to inform of any missing files.
        """
        self.data_path = data_path

        if patients is None:
            self.patients = sorted(os.listdir(data_path))
        else:
            self.patients = list(patients)

        if shuffle_all:
            np.random.shuffle(self.patients)

        self.augment = augment
        self.batch_size = batch_size
        self.shuffle_all = shuffle_all
        self.shuffle_batch = shuffle_batch
        self.retrieve_seg = retrieve_seg
        self.compressed_files = compressed_files
        self.verbose = verbose

    def __len__(self) -> int:
        return int(np.ceil(len(self.patients) / self.batch_size))

    def __getitem__(self, item: int) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Gets batch at index "item"
        :param item: index of batch to load.
        :return: tuple of (data, labels) for batch arrays.
        """
        if self.batch_size * (item + 1) >= len(self.patients):
            batch_patients = self.patients[self.batch_size * item:]
        else:
            batch_patients = self.patients[self.batch_size * item:self.batch_size * (item + 1)]

        if self.shuffle_batch:
            np.random.shuffle(batch_patients)

        data_batch = []
        if self.retrieve_seg:
            seg_batch = []

        for patient in batch_patients:
            patient_dir = os.path.join(self.data_path, patient)
            if self.verbose:
                print(f"Loading patient {patient}")
            try:
                data = []
                for scan_type in SCAN_TYPES:
                    scan = nib.load(os.path.join(patient_dir,
                                                 f"{patient}_{scan_type}.nii{'.gz' if self.compressed_files else ''}")
                                    ).get_fdata()
                    data.append(scan)

                data = np.stack(data, axis=-1)
                if self.retrieve_seg:
                    seg = nib.load(os.path.join(patient_dir,
                                                f"{patient}_seg.nii{'.gz' if self.compressed_files else ''}")
                                   ).get_fdata()

                if self.augment and self.retrieve_seg:
                    data, seg = apply_augmentation(data, seg)

                data_batch.append(data)
                if self.retrieve_seg:
                    seg_batch.append(seg)
            except FileNotFoundError as e:
                if self.verbose:
                    print(f"Missing file for patient {patient}: {e.filename}")

        data_batch = np.stack(data_batch)

        if self.retrieve_seg:
            seg_batch = np.stack(seg_batch)
            return data_batch, seg_batch
        return data_batch
