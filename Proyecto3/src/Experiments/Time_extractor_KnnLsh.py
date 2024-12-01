import os
from tqdm import tqdm
import numpy as np
from Experiments.Experiments_local import process_image_opencv
from Searching.KnnLsh import LSH_wrapper
import time
import csv

def run_knn_lsh_experiment():
    directory_path = os.path.join("F:/", "DB2_Proyect", "portraits")
    
    features_path = os.getcwd()
    index = 'data/index_dict_color_cv.npz'
    vector = 'data/feature_vector_color_cv.npy'
    data_dict = np.load(os.path.join(features_path,index), allow_pickle=True)
    index_dict = data_dict['arr_0'].item()
    feature_vector = np.load(os.path.join(features_path,vector))
    
    size = len(feature_vector)
    sizes = np.linspace(1000, size, num=10, dtype=int)
    
    query_path = os.path.join(directory_path, "3504480.jpg")
    _, query = process_image_opencv(query_path)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    query = query.astype('float32')
    query_experiment_index = index_dict["3504480"]
    times = np.zeros(10)
    
    #Parametros para LSH
    inverted_dict = {index:name for name, index in index_dict.items()}
    d = len(feature_vector[0])
    nbits = 512

    for idx, sample_size in tqdm(enumerate(sizes)):
        # Crear un nuevo indexador LSH para cada tamaño de muestra
        lsh_indexing = LSH_wrapper(dimension=d, nbits=nbits, directory_path=directory_path)
        
        # Preparar los features para el tamaño de muestra actual
        if sample_size < query_experiment_index:
            features = np.concatenate([
                feature_vector[:sample_size],
                feature_vector[query_experiment_index:query_experiment_index+1]
            ]).astype('float32')
        else:
            features = feature_vector[:sample_size].astype('float32')
        
        # Indexar los features
        lsh_indexing.Indexing(features)
        
        # Medir tiempo de búsqueda
        start = time.time()
        results = lsh_indexing.Knn_lsh(query, k=8, inverted_dict=inverted_dict)
        times[idx] = time.time() - start
    
    def save_timing_data(sizes, times, output_path):
        output_file = os.path.join(output_path, 'times_knn_lsh.csv')
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['sizes_knn_lsh', 'times_knn_lsh'])
            for size, time_value in zip(sizes, times):
                writer.writerow([size, time_value])
    
    output_path = os.path.join(os.getcwd(), "src", "Experiments", "times")
    save_timing_data(sizes, times, output_path)
    print("Guardado en ...", output_path)

if __name__ == "__main__":
    run_knn_lsh_experiment()