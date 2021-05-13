import os

import nibabel as nib
import tensorflow as tf

from definitions import RESULTS_PATH, DATA_PATH, PREPROCESSED_DATA_PATH
from postprocessing.postprocessing_pipeline import postprocess_segmentation
from preprocessing.preprocessing_pipeline import preprocessing_pipeline
from training.dataloader import BraTSDataLoader
from training.loss import dice_loss
from training.metrics import dice_etc, dice_wt, dice_tc, weighted_dice_score

if __name__ == '__main__':

    model_path = os.path.join(RESULTS_PATH, "")
    test_data_path = os.path.join(DATA_PATH, "")
    preprocessed_test_data_path = os.path.join(PREPROCESSED_DATA_PATH, "")
    predictions_path = os.path.join(PREPROCESSED_DATA_PATH, "")

    print("Preprocessing data...")

    preprocessing_pipeline(test_data_path, preprocessed_test_data_path)

    print("Loading model...")

    model = tf.keras.models.load_model(model_path, custom_objects={"dice_loss": dice_loss,
                                                                   "dice_wt": dice_wt,
                                                                   "dice_tc": dice_tc,
                                                                   "dice_etc": dice_etc,
                                                                   "weighted_dice_score": weighted_dice_score})

    print("Evaluating...")

    patients = os.listdir(preprocessed_test_data_path)

    data_loader = BraTSDataLoader(preprocessed_test_data_path, augment=False, patients=patients,
                                  shuffle_all=False, shuffle_batch=False, batch_size=1,
                                  verbose=True)

    predictions = model.predict(data_loader)

    print("Post-processing predictions...")

    for patient, prediction in zip(patients, predictions[0]):
        seg = postprocess_segmentation(prediction)
        nib.save(nib.Nifti1Image(seg, None), os.path.join(predictions_path, f"{patient}.nii.gz"))