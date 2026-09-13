import FBPConvNet.FBPConvNet as FBPConvNet
from Metric.Metrics import psnr_metric, ssim_metric
import phantoms.Dataset as Dataset

import os
import logging
import time

from tensorflow import keras
from keras import models
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.optimizers import Adam

from Utils.Loggers import Logger
from Utils.MetricLogger import MetricLogger


# ============================================================
# CONFIGURATION
# ============================================================

PROJECTION = 15

QUANT_OF_TRAIN_IMGS = 2000
QUANT_OF_TEST_IMGS = 500

EPOCHS = 100
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.1

LEARNING_RATE = 1e-4


# ============================================================
# PATHS
# ============================================================

# Google Drive
MAIN_PATH = "/content/drive/MyDrive/NEW_FBPCONVNET"

RESULTS_PATH = (
    MAIN_PATH + 
    f"/projection_{PROJECTION}"
)

CHECKPOINT_PATH = os.path.join(
    RESULTS_PATH,
    "checkpoints"
)

IMG_PATH = os.path.join(
    RESULTS_PATH,
    "imgs"
)

LOG_PATH = os.path.join(
    RESULTS_PATH,
    "logs"
)


# Dataset
X_TRAIN_PATH = MAIN_PATH + f"/Dataset/{PROJECTION}/Train"
Y_TRAIN_PATH = MAIN_PATH + "/Dataset/GroundTruth/Train"

X_TEST_PATH = MAIN_PATH + f"/Dataset/{PROJECTION}/Test"
Y_TEST_PATH = MAIN_PATH + "/Dataset/GroundTruth/Test"


# ============================================================
# TRAIN
# ============================================================

def _train() -> None:

    logging.info("Creating model...")

    model = FBPConvNet.fbpconvnet_model()

    model.summary()

    logging.info(
        "TensorFlow version: %s",
        tf.__version__
    )

    logging.info(
        "GPUs available: %s",
        tf.config.list_physical_devices("GPU")
    )

    logging.info(
        "Train images: %s",
        QUANT_OF_TRAIN_IMGS
    )

    logging.info(
        "Test images: %s",
        QUANT_OF_TEST_IMGS
    )

    logging.info(
        "Projection: %s",
        PROJECTION
    )

    logging.info(
        "Epochs: %s",
        EPOCHS
    )

    logging.info(
        "Batch size: %s",
        BATCH_SIZE
    )

    logging.info(
        "Validation split: %s",
        VALIDATION_SPLIT
    )

    logging.info(
        "Learning rate: %s",
        LEARNING_RATE
    )

    x_train, y_train, x_test, y_test = _get_dataset()

    _compile(model)

    start = time.time()

    history = _fit(
        model,
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT
    )

    end = time.time()

    training_time = end - start

    logging.info(
        "Training time: %.2f seconds",
        training_time
    )

    _save_training_info(
        history,
        training_time
    )

    _evaluate(
        model,
        x_test,
        y_test
    )


# ============================================================
# DATASET
# ============================================================

def _get_dataset():

    print("Getting dataset...")

    x_train, y_train = Dataset.load_full_dataset_X_n_Y(
        X_TRAIN_PATH,
        Y_TRAIN_PATH,
        PROJECTION
    )

    x_test, y_test = Dataset.load_full_dataset_X_n_Y(
        X_TEST_PATH,
        Y_TEST_PATH,
        PROJECTION
    )

    print("TRAIN dataset size:", len(x_train))
    print("TEST dataset size:", len(x_test))

    print("X train shape:", x_train.shape)
    print("Y train shape:", y_train.shape)

    print("X test shape:", x_test.shape)
    print("Y test shape:", y_test.shape)

    print(
        "X train range:",
        x_train.min(),
        x_train.max()
    )

    print(
        "Y train range:",
        y_train.min(),
        y_train.max()
    )

    return (
        x_train,
        y_train,
        x_test,
        y_test
    )


# ============================================================
# COMPILE
# ============================================================

def _compile(model: models.Model) -> None:

    print("Compiling...")

    model.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="mse",
        metrics=[
            keras.metrics.MeanSquaredError(
                name="mse"
            ),
            psnr_metric,
            ssim_metric
        ]
    )


# ============================================================
# CALLBACKS
# ============================================================

def _get_checkpoints() -> list:

    print("Creating checkpoints...")

    os.makedirs(
        CHECKPOINT_PATH,
        exist_ok=True
    )

    os.makedirs(
        LOG_PATH,
        exist_ok=True
    )

    checkpoint_best = ModelCheckpoint(
        filepath=os.path.join(
            CHECKPOINT_PATH,
            "best_model.keras"
        ),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    csv_logger = CSVLogger(
        os.path.join(
            LOG_PATH,
            "training_history.csv"
        ),
        append=False
    )

    return [
        checkpoint_best,
        early_stop,
        reduce_lr,
        csv_logger,
        MetricLogger()
    ]


# ============================================================
# FIT
# ============================================================

def _fit(
    model: models.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    validation_split: float
):

    print("Fitting model...")

    print("Epochs:", epochs)
    print("Batch size:", batch_size)
    print("Validation split:", validation_split)

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=_get_checkpoints(),
        verbose=1
    )

    return history


# ============================================================
# SAVE TRAINING INFORMATION
# ============================================================

def _save_training_info(
    history: keras.callbacks.History,
    training_time: float
) -> None:

    os.makedirs(
        IMG_PATH,
        exist_ok=True
    )

    os.makedirs(
        LOG_PATH,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Training curve
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        history.history["loss"],
        label="train"
    )

    plt.plot(
        history.history["val_loss"],
        label="validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()

    plt.savefig(
        os.path.join(
            IMG_PATH,
            "training_curve.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    # --------------------------------------------------------
    # Training history TXT
    # --------------------------------------------------------

    history_path = os.path.join(
        LOG_PATH,
        "training_history.txt"
    )

    with open(
        history_path,
        "w"
    ) as f:

        epochs_completed = len(
            history.history["loss"]
        )

        for epoch in range(
            epochs_completed
        ):

            f.write(
                f"Epoch {epoch + 1}\n"
            )

            for metric in history.history:

                value = history.history[
                    metric
                ][epoch]

                f.write(
                    f"{metric}: {value}\n"
                )

            f.write("\n")

        f.write(
            f"Training time: "
            f"{training_time:.2f} seconds\n"
        )

        f.write(
            f"Epochs completed: "
            f"{epochs_completed}\n"
        )

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    config_path = os.path.join(
        LOG_PATH,
        "training_config.txt"
    )

    with open(
        config_path,
        "w"
    ) as f:

        f.write(
            f"Projection: {PROJECTION}\n"
        )

        f.write(
            f"Train images: "
            f"{QUANT_OF_TRAIN_IMGS}\n"
        )

        f.write(
            f"Test images: "
            f"{QUANT_OF_TEST_IMGS}\n"
        )

        f.write(
            f"Validation split: "
            f"{VALIDATION_SPLIT}\n"
        )

        f.write(
            f"Epochs maximum: "
            f"{EPOCHS}\n"
        )

        f.write(
            f"Epochs completed: "
            f"{len(history.history['loss'])}\n"
        )

        f.write(
            f"Batch size: "
            f"{BATCH_SIZE}\n"
        )

        f.write(
            f"Learning rate: "
            f"{LEARNING_RATE}\n"
        )

        f.write(
            f"Optimizer: Adam\n"
        )

        f.write(
            f"Loss: MSE\n"
        )

        f.write(
            f"EarlyStopping patience: 5\n"
        )

        f.write(
            f"ReduceLROnPlateau patience: 3\n"
        )

        f.write(
            f"Training time: "
            f"{training_time:.2f} seconds\n"
        )


# ============================================================
# EVALUATE
# ============================================================

def _evaluate(
    model: models.Model,
    x_test: np.ndarray,
    y_test: np.ndarray
) -> None:

    print("Evaluating model...")

    results = model.evaluate(
        x_test,
        y_test,
        verbose=1
    )

    print("\nTest results:")

    results_path = os.path.join(
        LOG_PATH,
        "test_results.txt"
    )

    with open(
        results_path,
        "w"
    ) as f:

        f.write(
            f"Projection: {PROJECTION}\n\n"
        )

        for name, value in zip(
            model.metrics_names,
            results
        ):

            print(
                f"{name}: {value:.6f}"
            )

            f.write(
                f"{name}: {value:.6f}\n"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    Logger()

    logging.info(
        "Starting training..."
    )

    _train()