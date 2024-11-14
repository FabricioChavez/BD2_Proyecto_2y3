
import os
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from skimage import io, color
import numpy as np
import cv2

def extract_global_lbp_features(image, P=8, R=1, method='uniform', bins=4):
    lbp_features = []
    for channel in range(3):  # RGB
        lbp = local_binary_pattern(image[:, :, channel], P=P, R=R, method=method)
        # Calcular histograma global
        hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  # Normalización
        lbp_features.extend(hist)
    return np.array(lbp_features)  # 3 canales * bins dimensiones

def extract_global_color_histogram(image, bins=4):
    color_hist = []
    for channel in range(3):  # RGB
        hist, _ = np.histogram(image[:, :, channel], bins=bins, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  # Normalización
        color_hist.extend(hist)
    return np.array(color_hist)  # 3 canales * bins dimensiones

def extract_global_hog_features_opencv(gray_image, orientations=6, pixels_per_cell=(64, 64), cells_per_block=(1, 1)):
    hog = cv2.HOGDescriptor(_winSize=(gray_image.shape[1] // pixels_per_cell[1] * pixels_per_cell[1],
                                      gray_image.shape[0] // pixels_per_cell[0] * pixels_per_cell[0]),
                            _blockSize=(pixels_per_cell[1] * cells_per_block[1],
                                        pixels_per_cell[0] * cells_per_block[0]),
                            _blockStride=(pixels_per_cell[1], pixels_per_cell[0]),
                            _cellSize=pixels_per_cell,
                            _nbins=orientations)
    hog_features = hog.compute(gray_image).flatten()
    if len(hog_features) > orientations:
        hog_features = np.mean(hog_features.reshape(-1, orientations), axis=0)
    elif len(hog_features) < orientations:
        hog_features = np.pad(hog_features, (0, orientations - len(hog_features)), 'constant')
    return hog_features

def process_image_global(image_path, orientations=6, pixels_per_cell=(64, 64), cells_per_block=(1, 1), 
                         P=8, R=1, lbp_bins=4, color_bins=4):
    image = io.imread(image_path)
    
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"La imagen {image_path} no es RGB o no tiene 3 canales.")

    # Validar tamaño de la imagen (512x512)
    if image.shape[0] != 512 or image.shape[1] != 512:
        raise ValueError(f"La imagen {image_path} no tiene tamaño 512x512. Tamaño actual: {image.shape[:2]}")

    # Convertir a escala de grises
    gray_image = color.rgb2gray(image)
    gray_image_uint8 = (gray_image * 255).astype('uint8')

    # Extraer características globales
    hog_features = extract_global_hog_features_opencv(gray_image_uint8, orientations, pixels_per_cell, cells_per_block)
    lbp_features = extract_global_lbp_features(image, P=P, R=R, method='uniform', bins=lbp_bins)
    color_hist = extract_global_color_histogram(image, bins=color_bins)

    # Concatenar todas las características
    feature_vector = np.concatenate([hog_features, lbp_features, color_hist])

    # Validar tamaño del vector
    if feature_vector.shape[0] > 20:
        feature_vector = feature_vector[:20]
    elif feature_vector.shape[0] < 20:
        feature_vector = np.pad(feature_vector, (0, 20 - feature_vector.shape[0]), 'constant')

    return image_path, feature_vector


