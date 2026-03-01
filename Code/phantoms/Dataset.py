import numpy as np
from PIL import Image
import random

import os
from skimage.io import imread, imsave
from skimage.transform import radon, iradon, resize
from skimage.color import rgb2gray

from tensorflow import keras
from keras.utils import load_img, img_to_array

def generate_custom_data_set(quant_of_phantoms : int, x_path : str, y_path, projections : list[int]) -> None:
    
    ## Generating x
    
    for i in range(quant_of_phantoms):
        print(i)

        phantom = generate_random_phantom(size=512, num_ellipses=random.randrange(2, 14), seed=None)

        img_uint8 = ((phantom - phantom.min()) /
                     (phantom.max() - phantom.min()) * 255).astype(np.uint8)

        Image.fromarray(img_uint8).save(y_path + "/" + str(i) + ".png")

    
    ## Generating y
    
    for n_proj in projections:
        os.makedirs(os.path.join(x_path, str(n_proj)), exist_ok=True)

    image_files = [f for f in os.listdir(y_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

    for img_name in image_files:
        img_path = os.path.join(y_path, img_name)
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

            out_path = os.path.join(x_path, str(n_proj), img_name)
            imsave(out_path, reconstruction_uint8)


def load_dataset_X_n_Y(x_path : str, y_path : str) -> tuple[np.ndarray, np.ndarray]:
    X = []
    Y = []

    x_files = sorted(os.listdir(x_path))
    y_files = sorted(os.listdir(y_path))

    for x_file, y_file in zip(x_files, y_files):
        x_img = load_img(os.path.join(x_path, x_file), color_mode="grayscale")
        y_img = load_img(os.path.join(y_path, y_file), color_mode="grayscale")

        x_img = img_to_array(x_img) / 255.0
        y_img = img_to_array(y_img) / 255.0

        X.append(x_img)
        Y.append(y_img)

    return np.array(X), np.array(Y)

def load_full_dataset_X_n_Y(x_path : str, y_path : str, projections : list[int]) -> tuple[np.ndarray, np.ndarray]:
    X_total = []
    Y_total = []

    for p in projections:
        x, y = load_dataset_X_n_Y(x_path + f"/{p}", y_path)
        X_total.append(x)
        Y_total.append(y)

    return np.concatenate(X_total, axis=0), np.concatenate(Y_total, axis=0)

def generate_random_phantom(size : int = 256, num_ellipses : int = 10, seed : int | None = None) -> np.ndarray:
    """
    Gera um phantom aleatório baseado em elipses.
    
    Parâmetros:
        size (int): tamanho da imagem (size x size)
        num_ellipses (int): quantidade de elipses
        seed (int): define a semente aleatória (para reprodutibilidade)
    
    Retorna:
        np.ndarray: imagem phantom normalizada entre [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Cria grade normalizada (-1 a 1)
    y, x = np.mgrid[-1:1:size*1j, -1:1:size*1j]
    img = np.zeros((size, size), dtype=float)
    
    for _ in range(num_ellipses):
        # Parâmetros aleatórios da elipse
        cx, cy = np.random.uniform(-0.5, 0.5, 2)         # centro
        a, b = np.random.uniform(0.05, 0.4, 2)           # eixos
        angle = np.random.uniform(0, np.pi)              # rotação
        intensity = np.random.uniform(-1.0, 1.0)         # brilho
        
        # Equação da elipse rotacionada
        x_rot = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
        y_rot = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
        
        mask = (x_rot / a)**2 + (y_rot / b)**2 <= 1
        img[mask] += intensity

    # Normaliza entre 0 e 1
    img -= img.min()
    img /= img.max() if img.max() != 0 else 1
    
    return img