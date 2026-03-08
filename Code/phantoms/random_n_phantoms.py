import numpy as np
from PIL import Image
import random

def generate_random_phantom(size=256, num_ellipses=10, seed=None):
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

for i in range(10): ##? GERAR N PHANTOMS ALEATORIOS

    phantom = generate_random_phantom(size=512, num_ellipses=random.randrange(2, 14), seed=None)

    img_uint8 = ((phantom - phantom.min()) /
                 (phantom.max() - phantom.min()) * 255).astype(np.uint8)

    Image.fromarray(img_uint8).save("dataset/input/" + str(i) + ".png")