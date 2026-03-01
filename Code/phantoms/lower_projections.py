import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import radon, iradon, resize
from skimage.color import rgb2gray

input_dir = "dataset/input"
output_dir = "dataset/output"

projections = [15, 30, 45, 60, 90, 120, 150, 180]

for n_proj in projections:
    os.makedirs(os.path.join(output_dir, str(n_proj)), exist_ok=True)

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

for img_name in image_files:
    img_path = os.path.join(input_dir, img_name)
    print(f"Processando {img_name}...")

    image = imread(img_path)
    if image.ndim == 3:
        image = rgb2gray(image)
    image = resize(image, (512, 512))

    for n_proj in projections:
        theta = np.linspace(0., 180., n_proj, endpoint=False)

        # Forward projection (Radon)
        sinogram = radon(image, theta=theta, circle=False)

        # Reconstrução (Inverse Radon)
        reconstruction = iradon(sinogram, theta=theta, circle=False)

        reconstruction = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min())

        reconstruction_uint8 = (reconstruction * 255).astype(np.uint8)

        out_path = os.path.join(output_dir, str(n_proj), img_name)
        imsave(out_path, reconstruction_uint8)
