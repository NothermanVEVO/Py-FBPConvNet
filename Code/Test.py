import FBPConvNet.FBPConvNet as FBPConvNet

import numpy as np

print("Creating model...")

model = FBPConvNet.fbpconvnet_model()

print("Creating image...")

# cria uma imagem de teste
x = np.random.rand(1, 512, 512, 1)

print("Predicting image...")

y = model.predict(x)

print("Min:", y.min())
print("Max:", y.max())