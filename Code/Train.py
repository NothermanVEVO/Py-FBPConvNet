import FBPConvNet.FBPConvNet as FBPConvNet
from Codes.Metric.Metrics import psnr_metric
import phantoms.Dataset as Dataset
import os
from tensorflow import keras
from keras import models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

QUANT_OF_TRAIN_IMGS = 4
X_TRAIN_PATH = "dataset/x_train"
Y_TRAIN_PATH = "dataset/y_train"

QUANT_OF_TEST_IMGS = 2
X_TEST_PATH = "dataset/x_test"
Y_TEST_PATH = "dataset/y_test"

PROJECTIONS = [15, 30, 45, 60, 90, 120, 150, 180]


def _train(generate_dataset : bool = False) -> None:
    model = FBPConvNet.fbpconvnet_model()

    model.summary()
    
    if generate_dataset: _generate_datasets()

    x_train, y_train, x_test, y_test = _get_dataset()
    _compile(model)
    _fit(model, x_train, y_train)
    _evaluate(model, x_test, y_test)
    _predict(model, x_test[0], y_test[0])

def _generate_datasets() -> None:
    print("Generating TRAIN dataset...")
    os.makedirs(X_TRAIN_PATH, exist_ok=True)
    os.makedirs(Y_TRAIN_PATH, exist_ok=True)
    Dataset.generate_custom_data_set(QUANT_OF_TRAIN_IMGS, X_TRAIN_PATH, Y_TRAIN_PATH, PROJECTIONS)

    print("Generating TEST dataset...")
    os.makedirs(X_TEST_PATH, exist_ok=True)
    os.makedirs(Y_TEST_PATH, exist_ok=True)
    Dataset.generate_custom_data_set(QUANT_OF_TEST_IMGS, X_TEST_PATH, Y_TEST_PATH, PROJECTIONS)

def _get_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("Getting dataset...")

    x_train, y_train = Dataset.load_full_dataset_X_n_Y(X_TRAIN_PATH, Y_TRAIN_PATH, PROJECTIONS)
    x_test, y_test = Dataset.load_full_dataset_X_n_Y(X_TEST_PATH, Y_TEST_PATH, PROJECTIONS)

    return x_train, y_train, x_test, y_test

def _compile(model : models.Model) -> None:
    print("Compiling...")

    model.compile(
        optimizer = "adam",
        loss = "mse",
        metrics = [keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError(), psnr_metric]
    )

def _get_checkpoints() -> list:
    print("Epoch checkpoint...")

    os.makedirs("checkpoints", exist_ok=True)

    checkpoint_epoch = ModelCheckpoint( ## Each epoch will be saved
        "checkpoints/checkpoint_epoch_{epoch:03d}.keras",
        save_freq="epoch"
    )

    print("Best checkpoint...")

    checkpoint_best = ModelCheckpoint( ## Save the best model
        filepath="checkpoints/best_model.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",                # "min" for loss, "max" for SSIM
        verbose=1
    )

    print("Early stopping...")

    early_stop = EarlyStopping( ## To avoid overfitting
        monitor="val_loss",
        patience=5,          # wait 5 epochs to improve
        restore_best_weights=True
    )

    print("ReduceLROnPlateau...")

    reduce_lr = ReduceLROnPlateau( ## Reduces the learning rate when the model stops improving
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    return [checkpoint_epoch, checkpoint_best, early_stop, reduce_lr]

def _fit(model : models.Model, x_train : np.ndarray, y_train : np.ndarray) -> None:
    print("Fitting...")

    history = model.fit(
        x_train,
        y_train,
        epochs=2,
        batch_size=4,
        validation_split=0.2,
        callbacks=_get_checkpoints()
    )

    os.makedirs("imgs", exist_ok=True)

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["train", "val"])
    plt.savefig("imgs/training_curve.png", dpi=300)
    plt.show()

def _evaluate(model : models.Model, x_test : np.ndarray, y_test : np.ndarray) -> None:
    print("Evaluating...")

    results = model.evaluate(x_test, y_test, verbose=1)

    print("Results...")

    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value}")

def _predict(model : models.Model, x_test : Image, y_test : Image) -> None:
    print("Predicting...")

    pred = model.predict(x_test)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(x_test)
    plt.title("Input")

    plt.subplot(1,3,2)
    plt.imshow(y_test)
    plt.title("Ground Truth")

    plt.subplot(1,3,3)
    plt.imshow(pred[0])
    plt.title("Prediction")

    plt.savefig("imgs/model_prediction.png", dpi=300)

    plt.show()



if __name__ == "__main__":
    _train(generate_dataset=False)