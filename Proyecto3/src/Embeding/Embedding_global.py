import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
from skimage.feature import local_binary_pattern
from skimage import io, color
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler

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

def process_images_in_batches_global(images_paths, batch_size, orientations=6, pixels_per_cell=(64, 64), 
                                     cells_per_block=(1, 1), P=8, R=1, lbp_bins=4, color_bins=4):
    all_descriptors = {}
    total_images = len(images_paths)
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = []
        for i in range(0, len(images_paths), batch_size):
            batch_paths = images_paths[i:i + batch_size]
            futures.append(executor.submit(process_batch_global, batch_paths, orientations, pixels_per_cell, 
                                           cells_per_block, P, R, lbp_bins, color_bins))

        with tqdm(total=total_images, desc="Procesando imágenes") as pbar:
            for future in as_completed(futures):
                batch_descriptors = future.result()
                all_descriptors.update(batch_descriptors)
                pbar.update(len(batch_descriptors))

    return all_descriptors

def process_batch_global(batch_paths, orientations, pixels_per_cell, cells_per_block, P, R, lbp_bins, color_bins):
    batch_descriptors = {}
    for image_path in batch_paths:
        try:
            img_path, descriptor = process_image_global(image_path, orientations, pixels_per_cell, cells_per_block, 
                                                        P, R, lbp_bins, color_bins)
            if descriptor is not None:
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                batch_descriptors[image_name] = descriptor
        except Exception as e:
            print(f"Error procesando {image_path}: {e}")
    return batch_descriptors

def save_descriptors_npz(descriptors_dict, output_path):
    np.savez_compressed(output_path, **descriptors_dict)
    print(f"Todos los descriptores se han guardado en {output_path}")

if __name__ == "__main__":
    directory_path = os.path.join("F:/", "DB2_Proyect", "portraits")
    images_names = [name for name in os.listdir(directory_path) if name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images_paths = [os.path.join(directory_path, name) for name in images_names]

    path = "D:/Semestre_2024_2_CS/BD_2/Projects/Proyecto3"
    output_path = os.path.join(path, "data/descriptors_color_global_opencv.npz")

    batch_size = 500
    all_descriptors = process_images_in_batches_global(images_paths, batch_size)
    save_descriptors_npz(all_descriptors, output_path)
