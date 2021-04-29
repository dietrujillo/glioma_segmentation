import os

# GLOBAL PARAMETERS
RANDOM_SEED = 42
SCAN_TYPES = ("t1", "t2", "flair", "t1ce")
INPUT_DATA_SHAPE = (240, 240, 155)
PREPROCESSED_DATA_SHAPE = (200, 200, 144)

# FILE PATHS
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
PREPROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "preprocessed")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")

# PREPROCESSING & AUGMENTATION
NO_AUGMENTATION_PROBABILITY = 0.5
MAX_CROP_LIMIT = [(40, 196), (29, 222), (0, 148)]
CROP_LIMIT = [(20, 220), (20, 220), (0, 156)]
SEGMENTATION_CATEGORIES = [0., 1., 2., 4.]
SEGMENTATION_MERGE_DICT = {
    0: tuple(),  # Enhancing tumor core does not merge
    1: (0, 2),  # All of the tumor regions form the whole tumor
    2: (0,)  # The two innermost tumor regions form the tumor core
}

# TRAINING
DEFAULT_OPTIMIZER = "adam"
DEFAULT_LOSS = "categorical_crossentropy"
DEFAULT_EPOCHS = 30
BATCH_SIZE = 32
EARLY_STOPPING_PARAMS = {
    "monitor": "val_loss",
    "min_delta": 0,
    "patience": 1,
    "baseline": None,
    "restore_best_weights": True
}
LOSS_WEIGHTS = [0.1, 0.4, 0.3, 0.2]
