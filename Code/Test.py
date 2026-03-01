import FBPConvNet.FBPConvNet as FBPConvNet
from Codes.Loss.Losses import PSNRLoss
import phantoms.Dataset as Dataset
import numpy as np


model = FBPConvNet.fbpconvnet_model()

model.summary()

X_PATH = "dataset/x_train"
Y_PATH = "dataset/y_train"

projections = [15, 30, 45, 60, 90, 120, 150, 180]

# Dataset.generate_custom_data_set(10, X_PATH, Y_PATH, projections)

X_total = []
Y_total = []

for p in projections:
    x, y = Dataset.load_dataset_X_n_Y(X_PATH, Y_PATH + f"/{p}")
    X_total.append(x)
    Y_total.append(y)

x_train = np.concatenate(X_total, axis=0)
y_train = np.concatenate(Y_total, axis=0)

print(x_train)
print(y_train)

print("Compiling...")

model.compile(
    optimizer = "adam",
    loss = PSNRLoss(),
    metrics = ["mse", "mae"]
)

print("Fitting...")

model.fit(
    x_train,
    y_train,
    epochs=25,
    batch_size=4,
    validation_split=0.3
)

print("Ok's")