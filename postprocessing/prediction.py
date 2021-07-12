import os

import numpy as np
import nibabel as nib
import tensorflow as tf
import dask
from dask.diagnostics import ProgressBar

from definitions import RESULTS_PATH, DATA_PATH, PREPROCESSED_DATA_PATH
from postprocessing.postprocessing_pipeline import postprocess_segmentation
from preprocessing.preprocessing_pipeline import preprocessing_pipeline
from training.dataloader import BraTSDataLoader
from training.loss import dice_loss
from training.metrics import dice_etc, dice_wt, dice_tc, weighted_dice_score


if __name__ == '__main__':

    model_path = os.path.join(RESULTS_PATH, "28_aggregator_no_bn_relu/model.tf")
    test_data_path = os.path.join(DATA_PATH, "MICCAI_BraTS2020_ValidationData")
    preprocessed_test_data_path = os.path.join(PREPROCESSED_DATA_PATH, "MICCAI_BraTS2020_ValidationData")
    predictions_path = os.path.join(PREPROCESSED_DATA_PATH, "MICCAI_BraTS2020_ValidationPredictions")

    os.makedirs(predictions_path, exist_ok=True)

    print("Preprocessing data...")

    preprocessing_pipeline(test_data_path, preprocessed_test_data_path, process_seg=False)

    print("Loading model...")

    model = tf.keras.models.load_model(model_path, custom_objects={"dice_loss": dice_loss,
                                                                   "dice_wt": dice_wt,
                                                                   "dice_tc": dice_tc,
                                                                   "dice_etc": dice_etc,
                                                                   "weighted_dice_score": weighted_dice_score})

    print("Evaluating...")

    patients = sorted(os.listdir(preprocessed_test_data_path))

    data_loader = BraTSDataLoader(preprocessed_test_data_path, augment=False, patients=patients,
                                  retrieve_seg=False, compressed_files=False,
                                  shuffle_all=False, shuffle_batch=False, batch_size=1,
                                  verbose=True)

    predictions = model.predict(data_loader)

    print("Post-processing predictions...")

    header = nib.load(os.path.join(DATA_PATH, "MICCAI_BraTS2020_TrainingData/HGG/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz")).header

    @dask.delayed
    def distributed_postprocessing(patient, prediction):
        seg = postprocess_segmentation(prediction)
        image = nib.Nifti1Image(seg, None, header=header)
        nib.save(image, os.path.join(predictions_path, f"{patient}.nii.gz"))

    delayed_ops = []
    for patient, prediction in zip(patients, predictions):
        delayed_ops.append(distributed_postprocessing(patient, prediction))

    with ProgressBar():
        dask.compute(delayed_ops)
