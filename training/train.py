import gc
import os
from typing import Union, AnyStr, Iterable, Dict, Any, Tuple, Callable

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
from training.dataloader import BraTSDataLoader
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
          loss: Union[AnyStr, tf.keras.losses.Loss, Callable] = DEFAULT_LOSS,
          metrics: Iterable[tf.keras.metrics.Metric] = METRICS,
          epochs: int = DEFAULT_EPOCHS,
          batch_size: int = BATCH_SIZE,
          early_stopping_params: Dict[AnyStr, Any] = EARLY_STOPPING_PARAMS,
          random_state: int = RANDOM_SEED) \
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
    :param random_state: seed for the random number generator.
    :return: History object with training results.
    """
    setup_gpu()
    if random_state is not None:
        init_random_seed(random_state)

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

    history = model.fit(BraTSDataLoader(data_path, augment=False, batch_size=batch_size), epochs=epochs,
                        validation_data=BraTSDataLoader(val_data_path, augment=False, batch_size=batch_size),
                        batch_size=batch_size, validation_batch_size=batch_size, callbacks=callbacks)

    print(f"Training of model {training_id} finished.")
    cleanup()

    model.save(os.path.join(results_dir, "model.tf"), save_format="tf")

    return history
