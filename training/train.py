import gc
import os
from typing import Union, AnyStr, Iterable, Dict, Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, CSVLogger
from tensorflow.keras.models import Model

from definitions import (
    DATA_PATH, RESULTS_PATH,
    PREPROCESSED_DATA_SHAPE, SCAN_TYPES,
    DEFAULT_OPTIMIZER, DEFAULT_LOSS, DEFAULT_EPOCHS,
    EARLY_STOPPING_PARAMS, BATCH_SIZE,
    RANDOM_SEED
)
from training.dataloader import data_loader
from training.metrics import METRICS


def init_random_seed(random_state: int = RANDOM_SEED):
    """
    Initializes all random number generators with a random seed.
    :param random_state: Random seed.
    :return: None.
    """
    np.random.seed(random_state)
    tf.random.set_seed(random_state)


def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    print("Setting up GPUs...")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
            raise
    print("GPUs are ready.")


def limit_mem():
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=cfg))


def cleanup():
    gc.collect()
    sess = tf.compat.v1.keras.backend.get_session()
    sess.close()


def train(training_id: AnyStr,
          model: Model,
          data_path: str = DATA_PATH,
          val_data_path: str = DATA_PATH,
          input_shape: Tuple[int] = (None, *PREPROCESSED_DATA_SHAPE, len(SCAN_TYPES)),
          optimizer: Union[AnyStr, tf.keras.optimizers.Optimizer] = DEFAULT_OPTIMIZER,
          loss: Union[AnyStr, tf.keras.losses.Loss] = DEFAULT_LOSS,
          metrics: Iterable[tf.keras.metrics.Metric] = METRICS,
          epochs: int = DEFAULT_EPOCHS,
          batch_size: int = BATCH_SIZE,
          early_stopping_params: Dict[AnyStr, Any] = EARLY_STOPPING_PARAMS) \
        -> tf.keras.callbacks.History:
    """
    Trains a model.
    :param training_id: identifier for the training run. Must be unique.
    :param model: model to train. Must be an instance of tf.keras.models.Model.
    :param data_path: path to training data.
    :param val_data_path: path to validation data.
    :param input_shape: input shape.
    :param optimizer: optimizer to use.
    :param loss: loss function to use.
    :param metrics: metrics to use.
    :param epochs: maximum number of epochs (early stopping enabled by default)
    :param batch_size: batch size.
    :param early_stopping_params: dict with params for tf.keras.callbacks.EarlyStopping.
    :return: History object with training results.
    """
    limit_mem()
    setup_gpu()
    if RANDOM_SEED is not None:
        init_random_seed(RANDOM_SEED)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.build(input_shape=input_shape)

    print(model.summary())

    results_dir = os.path.join(RESULTS_PATH, training_id)
    os.makedirs(results_dir, exist_ok=False)

    early_stopping = EarlyStopping(**early_stopping_params)
    tensorboard = TensorBoard(log_dir=os.path.join(results_dir, "tensorboard_logs"))
    csv_logger = CSVLogger(os.path.join(results_dir, "log.csv"))

    callbacks = [early_stopping, tensorboard, csv_logger]

    print(f"Start training of model {training_id}.")

    history = model.fit(data_loader(data_path, augment=False, batch_size=batch_size), epochs=epochs,
                        validation_data=data_loader(val_data_path, augment=False, batch_size=batch_size),
                        batch_size=batch_size, validation_batch_size=batch_size, callbacks=callbacks)

    print(f"Training of model {training_id} finished.")
    cleanup()

    model.save(os.path.join(results_dir, "model.tf"), save_format="tf")

    return history
