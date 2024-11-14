import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
from skimage.feature import local_binary_pattern
from skimage import io, color
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler

def extract_lbp_features(cell_image, P=8, R=1, method='uniform', bins=4):
    lbp_features = []
    for channel in range(3):  # RGB
        lbp = local_binary_pattern(cell_image[:, :, channel], P=P, R=R, method=method)
        # Calcular histograma
        hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
        # Normalizar
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        lbp_features.extend(hist)
    return np.array(lbp_features)  # 3 canales * 4 bins = 12

def extract_color_histogram(cell_image, bins=4):
    color_hist = []
    for channel in range(3):  # RGB
        hist, _ = np.histogram(cell_image[:, :, channel], bins=bins, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        color_hist.extend(hist)
    return np.array(color_hist)  # 3 canales * 4 bins = 12

def extract_hog_features_opencv(gray_cell, orientations=4, pixels_per_cell=(64, 64), cells_per_block=(1, 1)):
    winSize = (gray_cell.shape[1] // pixels_per_cell[1] * pixels_per_cell[1],
               gray_cell.shape[0] // pixels_per_cell[0] * pixels_per_cell[0])
    hog = cv2.HOGDescriptor(_winSize=winSize,
                            _blockSize=(pixels_per_cell[1]*cells_per_block[1],
                                        pixels_per_cell[0]*cells_per_block[0]),
                            _blockStride=(pixels_per_cell[1], pixels_per_cell[0]),
                            _cellSize=pixels_per_cell,
                            _nbins=orientations)
    hog_features = hog.compute(gray_cell)
    hog_features = hog_features.flatten()
    # Reducir a 4 dimensiones si es necesario
    if len(hog_features) > orientations:
        hog_features = np.mean(hog_features.reshape(-1, orientations), axis=0)
    elif len(hog_features) < orientations:
        hog_features = np.pad(hog_features, (0, orientations - len(hog_features)), 'constant')
    return hog_features

def process_image_opencv(image_path, orientations=4, pixels_per_cell=(64, 64), cells_per_block=(1, 1), 
                        P=8, R=1, lbp_bins=4, color_bins=4):
    image = io.imread(image_path)
    
    # Validar que la imagen sea RGB
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"La imagen {image_path} no es RGB o no tiene 3 canales.")
    
    # Validar tamaño de la imagen (512x512)
    if image.shape[0] != 512 or image.shape[1] != 512:
        raise ValueError(f"La imagen {image_path} no tiene tamaño 512x512. Tamaño actual: {image.shape[:2]}")
    
    # Dividir la imagen en una cuadrícula de 8x8 (64 celdas de 64x64)
    grid_size = 8
    cell_height = image.shape[0] // grid_size
    cell_width = image.shape[1] // grid_size
    
    feature_matrix = []
    
    for row in range(grid_size):
        for col in range(grid_size):
            # Definir las coordenadas de la celda
            start_y = row * cell_height
            end_y = (row + 1) * cell_height
            start_x = col * cell_width
            end_x = (col + 1) * cell_width
            
            # Extraer la celda
            cell = image[start_y:end_y, start_x:end_x]
            
            # Convertir a escala de grises para HOG
            gray_cell = color.rgb2gray(cell)
            gray_cell_uint8 = (gray_cell * 255).astype('uint8')  # Convertir a uint8 para OpenCV
            
            # Extraer HOG usando OpenCV
            hog_features = extract_hog_features_opencv(gray_cell_uint8, orientations, pixels_per_cell, cells_per_block)
            # Asegurar que HOG contribuye con 4 dimensiones
            if len(hog_features) != orientations:
                # Reducir dimensionalidad de HOG mediante promedio si es necesario
                hog_features = np.mean(hog_features.reshape(-1, orientations), axis=0)
            
            # Extraer LBP
            lbp_features = extract_lbp_features(cell, P=P, R=R, method='uniform', bins=lbp_bins)
            # Reducir LBP a 4 dimensiones
            if len(lbp_features) > 4:
                lbp_features = lbp_features[:4]  # Seleccionar las primeras 4 dimensiones
            elif len(lbp_features) < 4:
                lbp_features = np.pad(lbp_features, (0, 4 - len(lbp_features)), 'constant')
            
            # Extraer Histograma de Color Local
            color_hist = extract_color_histogram(cell, bins=color_bins)
            # Reducir color_hist a 8 dimensiones
            if len(color_hist) > 8:
                color_hist = color_hist[:8]  # Seleccionar las primeras 8 dimensiones
            elif len(color_hist) < 8:
                color_hist = np.pad(color_hist, (0, 8 - len(color_hist)), 'constant')
            
            # Concatenar HOG + LBP + Color Histogram
            feature_vector = np.concatenate([hog_features, lbp_features, color_hist])
            # Asegurar que el feature_vector tiene exactamente 16 dimensiones
            if feature_vector.shape[0] != 16:
                raise ValueError(f"El vector de características de la celda ({row}, {col}) no tiene 16 dimensiones. Tiene {feature_vector.shape[0]} dimensiones.")
            
            feature_matrix.append(feature_vector)
    
    feature_matrix = np.array(feature_matrix)  # Shape: (64, 16)
    
    return image_path, feature_matrix

def process_batch_opencv(batch_paths, orientations=4, pixels_per_cell=(64, 64), cells_per_block=(1, 1), 
                         P=8, R=1, lbp_bins=4, color_bins=4):
    batch_descriptors = {}
    for image_path in batch_paths:
        try:
            img_path, descriptor = process_image_opencv(image_path, orientations, pixels_per_cell, cells_per_block, 
                                                       P, R, lbp_bins, color_bins)
            if descriptor is not None:
                # Usar el nombre del archivo (sin extensión) como clave
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                batch_descriptors[image_name] = descriptor
        except Exception as e:
            print(f"Error procesando {image_path}: {e}")
    return batch_descriptors

def process_images_in_batches_opencv(images_paths, batch_size, orientations=4, pixels_per_cell=(64, 64), 
                                     cells_per_block=(1, 1), P=8, R=1, lbp_bins=4, color_bins=4):
    all_descriptors = {}
    total_images = len(images_paths)
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = []
        for i in range(0, len(images_paths), batch_size):
            batch_paths = images_paths[i:i + batch_size]
            # Enviar cada lote como una tarea
            futures.append(executor.submit(process_batch_opencv, batch_paths, orientations, pixels_per_cell, 
                                           cells_per_block, P, R, lbp_bins, color_bins))
        
        with tqdm(total=total_images, desc="Procesando imágenes") as pbar:
            for future in as_completed(futures):
                batch_descriptors = future.result()
                all_descriptors.update(batch_descriptors)
                pbar.update(len(batch_descriptors))
    
    return all_descriptors



def save_descriptors_npz(descriptors_dict, output_path):
    # Guardar todos los descriptores en un solo archivo .npz comprimido
    np.savez_compressed(output_path, **descriptors_dict)
    print(f"Todos los descriptores se han guardado en {output_path}")

def generate_feature_vectors(images_descriptors , path):
    index = 'data/index_dict_color_cv.npz'
    vector ='data/feature_vector_color_cv.npy'
    index_dict = {}
    # Crear una matriz de ceros con dimensiones (cantidad de imágenes, 128)
    feature_vectors = np.zeros((len(images_descriptors), 1024))
    
    # Iterar sobre cada descriptor de imagen
    for i, (names, matrix) in enumerate(tqdm(images_descriptors.items())):
        v = np.array(matrix)
        #print(v.shape)
        feature_vectors[i] = v.flatten()  # Asignar el vector aplanado a la fila correspondiente
        index_dict[names] = i
    np.savez(os.path.join(path,index), index_dict) 
    np.save(os.path.join(path,vector), feature_vectors)
    return index_dict, feature_vectors



if __name__ == "__main__":
    directory_path = os.path.join("F:/", "DB2_Proyect", "portraits")
    images_names = [name for name in os.listdir(directory_path) if name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images_paths = [os.path.join(directory_path, name) for name in images_names]
    
    # Ruta de salida para el archivo .npz final
    path = "D:/Semestre_2024_2_CS/BD_2/Projects/Proyecto3"
    output_path = os.path.join(path, "data/descriptors_color_2_opencv.npz")

    # Procesar imágenes en lotes y guardar descriptores usando OpenCV para HOG
    batch_size = 500  # Ajusta el tamaño de lote a 100
    all_descriptors = process_images_in_batches_opencv(images_paths, batch_size)
    save_descriptors_npz(all_descriptors, output_path)
