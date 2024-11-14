from scipy.io import loadmat
from tqdm import tqdm
import os

path = os.path.join(os.getcwd(), "data")
print(f"Total archivos en el directorio: {len(os.listdir(path))}")

def load_descriptors_from_mat(directory_path):
    descriptors = {}
    images = os.listdir(directory_path)

    # Recorre todos los archivos .mat en el directorio especificado
    for filename in tqdm(images):
        if filename.endswith(".mat"):
            file_path = os.path.join(directory_path, filename)
            try:
                # Carga el archivo .mat
                mat_data = loadmat(file_path)
                
                # Extrae la matriz de descriptores (asumiendo que tiene un nombre conocido)
                key_name = next((key for key in mat_data.keys() if not key.startswith("__")), None)
                
                if key_name:
                    descriptors[filename] = mat_data[key_name]
                else:
                    print(f"Warning: No descriptor found in {filename}")
            
            except Exception as e:
                # Captura y muestra cualquier error durante la carga del archivo
                print(f"Error loading {filename}: {e}")

    return descriptors

# Cargar descriptores
images_descriptors = load_descriptors_from_mat(path)
print(f"Total descriptores cargados: {len(images_descriptors)}")