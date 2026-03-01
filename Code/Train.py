import FBPConvNet.FBPConvNet as FBPConvNet
from Codes.Metric.Metrics import psnr_metric
import phantoms.Dataset as Dataset
import os
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

model = FBPConvNet.fbpconvnet_model()

model.summary()

X_TRAIN_PATH = "dataset/x_train"
Y_TRAIN_PATH = "dataset/y_train"

X_TEST_PATH = "dataset/x_test"
Y_TEST_PATH = "dataset/y_test"

projections = [15, 30, 45, 60, 90, 120, 150, 180]

print("Generating TRAIN dataset...")
Dataset.generate_custom_data_set(4, X_TRAIN_PATH, Y_TRAIN_PATH, projections)

print("Generating TEST dataset...")
Dataset.generate_custom_data_set(2, X_TEST_PATH, Y_TEST_PATH, projections)

print("Getting dataset...")

x_train, y_train = Dataset.load_full_dataset_X_n_Y(X_TRAIN_PATH, Y_TRAIN_PATH, projections)
x_test, y_test = Dataset.load_full_dataset_X_n_Y(X_TEST_PATH, Y_TEST_PATH, projections)

print("x_train min/max:", x_train.min(), x_train.max())
print("y_train min/max:", y_train.min(), y_train.max())

print("Compiling...")

model.compile(
    optimizer = "adam",
    loss = "mse",
    metrics = [keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError(), psnr_metric]
)

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

print("Fitting...")

history = model.fit(
    x_train,
    y_train,
    epochs=2,
    batch_size=4,
    validation_split=0.2,
    callbacks=[checkpoint_epoch, checkpoint_best, early_stop, reduce_lr]
)

os.makedirs("imgs", exist_ok=True)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["train", "val"])
plt.savefig("imgs/training_curve.png", dpi=300)
plt.show()

print("Evaluating...")

results = model.evaluate(x_test, y_test, verbose=1)

print("Results...")

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value}")

print("Predicting...")

pred = model.predict(x_test)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(x_test[0])
plt.title("Input")

plt.subplot(1,3,2)
plt.imshow(y_test[0])
plt.title("Ground Truth")

plt.subplot(1,3,3)
plt.imshow(pred[0])
plt.title("Prediction")

plt.savefig("imgs/model_prediction.png", dpi=300)

plt.show()