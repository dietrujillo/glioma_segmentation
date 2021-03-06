import gc
import json
import os
from typing import Union, AnyStr, Iterable, Dict, Any, Tuple, Callable, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, CSVLogger
from tensorflow.keras.models import Model

from definitions import (
    DATA_PATH, RESULTS_PATH,
    PREPROCESSED_DATA_SHAPE, SCAN_TYPES,
    DEFAULT_OPTIMIZER, DEFAULT_LOSS, DEFAULT_EPOCHS,
    EARLY_STOPPING_PARAMS, BATCH_SIZE,
    COMPUTE_DEVICES, RANDOM_SEED
)
from models.unet import UNet
from models.model_aggregator import ModelAggregator
from training.dataloader import BraTSDataLoader
from training.loss import dice_loss
from training.metrics import METRICS, dice_wt, dice_tc, dice_etc, weighted_dice_score


def init_random_seed(random_state: int = RANDOM_SEED) -> None:
    """
    Initializes all random number generators with a random seed.
    :param random_state: Random seed.
    :return: None.
    """
    np.random.seed(random_state)
    tf.random.set_seed(random_state)


def setup_gpu() -> None:
    """
    Set TF_ALLOW_GPU_GROWTH to True.
    :return: None.
    """
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=cfg))


def cleanup() -> None:
    """
    Clean session and memory after training.
    :return: None.
    """
    gc.collect()
    sess = tf.compat.v1.keras.backend.get_session()
    sess.close()


def build_model(model_class: type,
                model_params=None,
                input_shape: Tuple[int] = (None, *PREPROCESSED_DATA_SHAPE, len(SCAN_TYPES)),
                optimizer: Union[AnyStr, tf.keras.optimizers.Optimizer] = DEFAULT_OPTIMIZER,
                loss: Union[AnyStr, tf.keras.losses.Loss, Callable] = DEFAULT_LOSS,
                metrics: Iterable[tf.keras.metrics.Metric] = METRICS) -> Model:
    """
    Builds a model, with optional model parallelism.
    :param model_class: Subclass of tf.keras.models.Model
    :param model_params: arguments for model_class initializer.
    :param input_shape: expected input shape for the model.
    :param optimizer: optimizer to use.
    :param loss: loss function to use.
    :param metrics: metrics to use.
    :return: built and compiled model.
    """

    if model_params is None:
        model_params = {}

    model = model_class(**model_params)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.build(input_shape=input_shape)

    print(model.summary())

    return model


def train(training_id: AnyStr,
          model: Model,
          data_path: str = DATA_PATH,
          val_data_path: str = DATA_PATH,
          epochs: int = DEFAULT_EPOCHS,
          batch_size: int = BATCH_SIZE,
          early_stopping_params: Dict[AnyStr, Any] = EARLY_STOPPING_PARAMS) -> None:
    """
    Trains a model.
    :param training_id: identifier for the training run. Must be unique.
    :param model: model to train. Must be an instance of tf.keras.models.Model.
    :param data_path: path to training data.
    :param val_data_path: path to validation data.
    :param epochs: maximum number of epochs (early stopping enabled by default)
    :param batch_size: batch size.
    :param early_stopping_params: dict with params for tf.keras.callbacks.EarlyStopping.
    :return: History object with training results.
    """

    results_dir = os.path.join(RESULTS_PATH, training_id)
    os.makedirs(results_dir, exist_ok=False)

    early_stopping = EarlyStopping(**early_stopping_params)
    tensorboard = TensorBoard(log_dir=os.path.join(results_dir, "tensorboard_logs"))
    csv_logger = CSVLogger(os.path.join(results_dir, "log.csv"))

    callbacks = [early_stopping, tensorboard, csv_logger]

    print(f"Start training of model {training_id}.")

    history = model.fit(BraTSDataLoader(data_path, augment=True, batch_size=batch_size, subdivide_sectors=False, verbose=False),
                        validation_data=BraTSDataLoader(val_data_path, augment=False, batch_size=batch_size),
                        epochs=epochs,
                        callbacks=callbacks)

    print(f"Training of model {training_id} finished.")
    cleanup()

    model.save(os.path.join(results_dir, "model.tf"), save_format="tf")
    with open(os.path.join(results_dir, "history.json"), "w") as file:
        json.dump(history.history, file, indent=4)



def main():
    u_net = build_model(
            ModelAggregator,
            model_params={
                "models": [
                    os.path.join(RESULTS_PATH, "14_split_augment_patience_20/model.tf"),
                    os.path.join(RESULTS_PATH, "16_inception/model.tf")
                ],
                "custom_layers": [
                    {
                        "dice_loss": dice_loss,
                        "dice_wt": dice_wt,
                        "dice_tc": dice_tc,
                        "dice_etc": dice_etc,
                        "weighted_dice_score": weighted_dice_score
                    },
                    {
                        "dice_loss": dice_loss,
                        "dice_wt": dice_wt,
                        "dice_tc": dice_tc,
                        "dice_etc": dice_etc,
                        "weighted_dice_score": weighted_dice_score
                    }
                ]
            },
            optimizer="nadam", loss=dice_loss)
            
    train("20_small_aggregator", u_net, 
          data_path="preprocessed/train",
          val_data_path="preprocessed/test",
          batch_size=BATCH_SIZE)

if __name__ == '__main__':

    init_random_seed(RANDOM_SEED)
    setup_gpu()

    if len(COMPUTE_DEVICES) > 1:
        distribution = tf.distribute.MirroredStrategy(COMPUTE_DEVICES)
        with distribution.scope():
            main()
    elif len(COMPUTE_DEVICES) == 1:
        with tf.device(COMPUTE_DEVICES[0]):
            main()
    else:
        raise ValueError("No compute device specified.")

        
