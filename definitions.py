import os

# GLOBAL PARAMETERS
RANDOM_SEED = 42
SCAN_TYPES = ("t1", "t2", "flair", "t1ce")
INPUT_DATA_SHAPE = (240, 240, 155)
PREPROCESSED_DATA_SHAPE = (128, 128, 96)

# FILE PATHS
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
PREPROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "preprocessed")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")

# PREPROCESSING & AUGMENTATION
MAX_CROP_LIMIT = [(40, 196), (29, 222), (0, 148)]
CROP_LIMIT = [(20, 220), (20, 220), (6, 150)]
CROP_SHAPE = (200, 200, 144)
RESIZE_SHAPE = (128, 128, 96)
SEGMENTATION_CATEGORIES = [0., 1., 2., 4.]
NO_AUGMENTATION_PROBABILITY = 0.5
ROTATION_MAX_DEGREES = (-2, 2)

# TRAINING
DEFAULT_COMPUTE_DEVICES = ["/device:GPU:0"]
DEFAULT_OPTIMIZER = "nadam"
DEFAULT_LOSS = "categorical_crossentropy"
DEFAULT_EPOCHS = 30
BATCH_SIZE = 4
EARLY_STOPPING_PARAMS = {
    "monitor": "val_weighted_dice_score",
    "min_delta": 0,
    "patience": 1,
    "baseline": None,
    "restore_best_weights": True
}
LOSS_WEIGHTS = [0.2, 0.35, 0.45]


