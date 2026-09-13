import FBPConvNet.FBPConvNet as FBPConvNet
from Metric.Metrics import psnr_metric, ssim_metric
import phantoms.Dataset as Dataset
import os
from tensorflow import keras
from keras import models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import logging
import time
from keras.optimizers import Adam

from Utils.Loggers import Logger
from Utils.MetricLogger import MetricLogger

PROJECTION = 15

QUANT_OF_TRAIN_IMGS = 2000
X_TRAIN_PATH = "Dataset/" + str(PROJECTION) + "/Train"
Y_TRAIN_PATH = "Dataset/GroundTruth/Train"

QUANT_OF_TEST_IMGS = 500
X_TEST_PATH = "Dataset/" + str(PROJECTION) + "/Test"
Y_TEST_PATH = "Dataset/GroundTruth/Test"

def _train() -> None:

    logging.info("Creating model...")

    model = FBPConvNet.fbpconvnet_model()

    model.summary()

    logging.info("TensorFlow version: %s", tf.__version__)
    logging.info("GPUs available: %s", tf.config.list_physical_devices("GPU"))
    logging.info("Train images: %s", QUANT_OF_TRAIN_IMGS)
    logging.info("Test images: %s", QUANT_OF_TEST_IMGS)
    logging.info("Projection: %s", PROJECTION)

    x_train, y_train, x_test, y_test = _get_dataset()

    _compile(model)

    start = time.time()

    _fit(model, x_train, y_train, epochs=100, batch_size=16, validation_split=0.1)

    end = time.time()

    logging.info("Training time: %.2f seconds", end - start)

    _evaluate(model, x_test, y_test)


def _get_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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

    return x_train, y_train, x_test, y_test


def _compile(model: models.Model) -> None:

    print("Compiling...")

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="mse",
        metrics=[
            keras.metrics.MeanSquaredError(name="mse"),
            psnr_metric,
            ssim_metric
        ]
    )


def _get_checkpoints() -> list:

    print("Creating checkpoints...")

    os.makedirs("checkpoints", exist_ok=True)

    ## REMOVED FOR NOW...
    #
    # checkpoint_epoch = ModelCheckpoint(
    #     "checkpoints/checkpoint_epoch_{epoch:03d}.keras",
    #     save_freq="epoch"
    # )

    checkpoint_best = ModelCheckpoint(
        filepath="checkpoints/best_model.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    return [checkpoint_best, early_stop, reduce_lr, MetricLogger()]


def _fit(
    model: models.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    validation_split: float
) -> None:

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
        callbacks=_get_checkpoints()
    )

    os.makedirs("imgs", exist_ok=True)

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["train", "val"])
    plt.savefig("imgs/training_curve.png", dpi=300)

    plt.show()

    os.makedirs("logs", exist_ok=True)

    with open("logs/training_history.txt", "w") as f:
        epochs = len(history.history["loss"])   

        for epoch in range(epochs):
            f.write(f"Epoch {epoch+1}\n")   

            for metric in history.history:
                value = history.history[metric][epoch]
                f.write(f"{metric}: {value}\n") 

            f.write("\n")


def _evaluate(model: models.Model, x_test: np.ndarray, y_test: np.ndarray) -> None:

    print("Evaluating model...")

    results = model.evaluate(x_test, y_test, verbose=1)

    print("Results:")

    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value}")


if __name__ == "__main__":

    Logger()

    logging.info("Starting training...")

    _train()