import FBPConvNet.FBPConvNet as FBPConvNet
from Metric.Metrics import psnr_metric, ssim_metric
import phantoms.Dataset as Dataset

import os
import csv
import json
import logging
import subprocess
import threading
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import models
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
)

from keras.optimizers import Adam

from Utils.Loggers import Logger


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

MAIN_PATH = "/content/drive/MyDrive/NEW_FBPCONVNET"

RESULTS_PATH = os.path.join(
    MAIN_PATH,
    f"projection_{PROJECTION}"
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


# ============================================================
# DATASET
# ============================================================

X_TRAIN_PATH = os.path.join(
    MAIN_PATH,
    "Dataset",
    str(PROJECTION),
    "Train"
)

Y_TRAIN_PATH = os.path.join(
    MAIN_PATH,
    "Dataset",
    "GroundTruth",
    "Train"
)

X_TEST_PATH = os.path.join(
    MAIN_PATH,
    "Dataset",
    str(PROJECTION),
    "Test"
)

Y_TEST_PATH = os.path.join(
    MAIN_PATH,
    "Dataset",
    "GroundTruth",
    "Test"
)


# ============================================================
# RESOURCE MONITOR
# ============================================================

class ResourceMonitor:

    def __init__(
        self,
        filepath: str,
        interval: float = 1.0
    ):

        self.filepath = filepath
        self.interval = interval

        self.running = False
        self.thread = None

        self.file = None
        self.writer = None

    def start(self):

        os.makedirs(
            os.path.dirname(self.filepath),
            exist_ok=True
        )

        self.file = open(
            self.filepath,
            "w",
            newline=""
        )

        self.writer = csv.writer(
            self.file
        )

        self.writer.writerow([
            "timestamp",
            "elapsed_seconds",
            "cpu_percent",
            "ram_used_gb",
            "ram_percent",
            "gpu_name",
            "gpu_util_percent",
            "gpu_memory_used_mb",
            "gpu_memory_total_mb"
        ])

        self.file.flush()

        self.start_time = time.time()

        self.running = True

        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )

        self.thread.start()

    def stop(self):

        self.running = False

        if self.thread is not None:
            self.thread.join()

        if self.file is not None:
            self.file.close()

    def _monitor(self):

        while self.running:

            elapsed = (
                time.time() -
                self.start_time
            )

            cpu_percent = 0.0
            ram_used_gb = 0.0
            ram_percent = 0.0

            try:

                import psutil

                cpu_percent = (
                    psutil.cpu_percent(
                        interval=None
                    )
                )

                memory = (
                    psutil.virtual_memory()
                )

                ram_used_gb = (
                    memory.used /
                    (1024 ** 3)
                )

                ram_percent = (
                    memory.percent
                )

            except Exception:
                pass

            gpu_name = ""
            gpu_util = 0.0
            gpu_memory_used = 0.0
            gpu_memory_total = 0.0

            try:

                result = (
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu="
                            "name,"
                            "utilization.gpu,"
                            "memory.used,"
                            "memory.total",
                            "--format="
                            "csv,noheader,nounits"
                        ],
                        encoding="utf-8"
                    )
                )

                values = [
                    value.strip()
                    for value in
                    result.strip().split(",")
                ]

                gpu_name = values[0]

                gpu_util = float(
                    values[1]
                )

                gpu_memory_used = float(
                    values[2]
                )

                gpu_memory_total = float(
                    values[3]
                )

            except Exception:
                pass

            self.writer.writerow([
                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                elapsed,
                cpu_percent,
                ram_used_gb,
                ram_percent,
                gpu_name,
                gpu_util,
                gpu_memory_used,
                gpu_memory_total
            ])

            self.file.flush()

            time.sleep(
                self.interval
            )


# ============================================================
# TRAIN
# ============================================================

def _train() -> None:

    _create_directories()

    logging.info(
        "Creating model..."
    )

    model = (
        FBPConvNet.fbpconvnet_model()
    )

    model.summary()

    _print_environment()

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

    x_train, y_train, x_test, y_test = (
        _get_dataset()
    )

    _compile(model)

    _save_configuration()

    resource_monitor = ResourceMonitor(
        os.path.join(
            LOG_PATH,
            "resource_usage.csv"
        ),
        interval=1.0
    )

    print("\nStarting training...\n")

    resource_monitor.start()

    start = time.time()

    try:

        history = _fit(
            model,
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT
        )

    finally:

        resource_monitor.stop()

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
# DIRECTORIES
# ============================================================

def _create_directories():

    os.makedirs(
        CHECKPOINT_PATH,
        exist_ok=True
    )

    os.makedirs(
        IMG_PATH,
        exist_ok=True
    )

    os.makedirs(
        LOG_PATH,
        exist_ok=True
    )


# ============================================================
# ENVIRONMENT
# ============================================================

def _print_environment():

    print("\nEnvironment:")
    print(
        "TensorFlow:",
        tf.__version__
    )

    print(
        "GPU:",
        tf.config.list_physical_devices(
            "GPU"
        )
    )

    try:

        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu="
                "name,memory.total",
                "--format="
                "csv,noheader"
            ],
            encoding="utf-8"
        )

        print(
            "GPU information:",
            result.strip()
        )

    except Exception:

        print(
            "GPU information unavailable"
        )


# ============================================================
# DATASET
# ============================================================

def _get_dataset():

    print(
        "\nGetting dataset..."
    )

    x_train, y_train = (
        Dataset.load_full_dataset_X_n_Y(
            X_TRAIN_PATH,
            Y_TRAIN_PATH,
            PROJECTION
        )
    )

    x_test, y_test = (
        Dataset.load_full_dataset_X_n_Y(
            X_TEST_PATH,
            Y_TEST_PATH,
            PROJECTION
        )
    )

    print(
        "TRAIN dataset size:",
        len(x_train)
    )

    print(
        "TEST dataset size:",
        len(x_test)
    )

    print(
        "X train shape:",
        x_train.shape
    )

    print(
        "Y train shape:",
        y_train.shape
    )

    print(
        "X test shape:",
        x_test.shape
    )

    print(
        "Y test shape:",
        y_test.shape
    )

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

def _compile(
    model: models.Model
) -> None:

    print(
        "\nCompiling..."
    )

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

def _get_checkpoints():

    print(
        "Creating callbacks..."
    )

    checkpoint_best = (
        ModelCheckpoint(
            filepath=os.path.join(
                CHECKPOINT_PATH,
                "best_model.keras"
            ),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        )
    )

    early_stop = (
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    )

    reduce_lr = (
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    )

    csv_logger = (
        CSVLogger(
            os.path.join(
                LOG_PATH,
                "training_history.csv"
            ),
            append=False
        )
    )

    return [
        checkpoint_best,
        early_stop,
        reduce_lr,
        csv_logger
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

    print(
        "\nFitting model..."
    )

    print(
        "Epochs:",
        epochs
    )

    print(
        "Batch size:",
        batch_size
    )

    print(
        "Validation split:",
        validation_split
    )

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
# SAVE CONFIGURATION
# ============================================================

def _save_configuration():

    path = os.path.join(
        LOG_PATH,
        "training_config.txt"
    )

    with open(path, "w") as f:

        f.write(
            f"Projection: {PROJECTION}\n"
        )

        f.write(
            f"Train images: "
            f"{QUANT_OF_TRAIN_IMGS}\n"
        )

        f.write(
            f"Validation split: "
            f"{VALIDATION_SPLIT}\n"
        )

        f.write(
            f"Validation images: "
            f"{int(QUANT_OF_TRAIN_IMGS * VALIDATION_SPLIT)}\n"
        )

        f.write(
            f"Training images: "
            f"{int(QUANT_OF_TRAIN_IMGS * (1 - VALIDATION_SPLIT))}\n"
        )

        f.write(
            f"Test images: "
            f"{QUANT_OF_TEST_IMGS}\n"
        )

        f.write(
            f"Epochs maximum: "
            f"{EPOCHS}\n"
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
            "Optimizer: Adam\n"
        )

        f.write(
            "Loss: MSE\n"
        )

        f.write(
            "Metrics: MSE, PSNR, SSIM\n"
        )

        f.write(
            "EarlyStopping: val_loss, patience=5\n"
        )

        f.write(
            "ReduceLROnPlateau: val_loss, patience=3, factor=0.5\n"
        )

        f.write(
            "ModelCheckpoint: best val_loss\n"
        )

        f.write(
            f"TensorFlow version: "
            f"{tf.__version__}\n"
        )

        f.write(
            "GPU information:\n"
        )

        try:

            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu="
                    "name,memory.total,"
                    "driver_version",
                    "--format="
                    "csv,noheader"
                ],
                encoding="utf-8"
            )

            f.write(
                result.strip() + "\n"
            )

        except Exception:

            f.write(
                "Unavailable\n"
            )


# ============================================================
# SAVE TRAINING INFORMATION
# ============================================================

def _save_training_info(
    history,
    training_time: float
):

    history_dict = (
        history.history
    )

    epochs_completed = len(
        history_dict["loss"]
    )

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    history_path = os.path.join(
        LOG_PATH,
        "training_history.txt"
    )

    with open(
        history_path,
        "w"
    ) as f:

        for epoch in range(
            epochs_completed
        ):

            f.write(
                f"Epoch {epoch + 1}\n"
            )

            for metric in history_dict:

                value = (
                    history_dict[
                        metric
                    ][epoch]
                )

                f.write(
                    f"{metric}: "
                    f"{value:.8f}\n"
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
    # GRAPHS
    # --------------------------------------------------------

    _plot_metric(
        history_dict,
        "loss",
        "val_loss",
        "Loss",
        "training_loss.png"
    )

    _plot_metric(
        history_dict,
        "mse",
        "val_mse",
        "MSE",
        "training_mse.png"
    )

    _plot_metric(
        history_dict,
        "psnr_metric",
        "val_psnr_metric",
        "PSNR",
        "training_psnr.png"
    )

    _plot_metric(
        history_dict,
        "ssim_metric",
        "val_ssim_metric",
        "SSIM",
        "training_ssim.png"
    )


# ============================================================
# PLOT
# ============================================================

def _plot_metric(
    history,
    train_name,
    validation_name,
    ylabel,
    filename
):

    if train_name not in history:
        return

    plt.figure()

    plt.plot(
        history[train_name],
        label="train"
    )

    if validation_name in history:

        plt.plot(
            history[validation_name],
            label="validation"
        )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        ylabel
    )

    plt.legend()

    plt.savefig(
        os.path.join(
            IMG_PATH,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# EVALUATE
# ============================================================

def _evaluate(
    model: models.Model,
    x_test: np.ndarray,
    y_test: np.ndarray
):

    print(
        "\nEvaluating model..."
    )

    results = model.evaluate(
        x_test,
        y_test,
        verbose=1
    )

    print(
        "\nTest results:"
    )

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
                f"{name}: "
                f"{value:.6f}\n"
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