import os
from tqdm import tqdm
import numpy as np
from Experiments.Experiments_ivf import process_image_opencv
from Searching.KnnIvF import normalize_query_descriptors, tf_idf_knn, key_generator, generate_inverted_index, calculate_idf, calculate_tf, prenormalization
import time
import csv

def run_knn_ivf_experiment():
    directory_path = os.path.join("F:/", "DB2_Proyect", "portraits")
    
    output_directory = os.path.join(os.getcwd(), "data")
    data_name = 'features_data_2.npz'
    path = os.path.join(output_directory, data_name)
    
    loaded_data = np.load(path, allow_pickle=True)
    total_features_index = loaded_data["total_features_index"].item()
    total_features = loaded_data["total_features"]
    
    loading_path = os.path.join(os.getcwd(), "data")
    size = 302652
    sizes = np.linspace(1000, size, num=10, dtype=int)
    times = np.zeros(10)
    
    query_path = os.path.join(directory_path, "10-luyjxyh.jpg")
    _, query = process_image_opencv(query_path)
    query_normalized = normalize_query_descriptors(query_matrix=query)
    
    for idx, sample_size in tqdm(enumerate(sizes)):
        visual_words_file = f"visual_words_500_v3_{sample_size}.npz"
        vwords_path = os.path.join(loading_path, visual_words_file)
        loaded_data = np.load(vwords_path, allow_pickle=True)
        visual_words = loaded_data["visual_words"]
        centroids = loaded_data["centroides"]
        
        inverted_index = generate_inverted_index(visual_words=visual_words, total_features_index=total_features_index)
        idf_vector = calculate_idf(inverted_index=inverted_index, N=sample_size)
        tf_dict = calculate_tf(inverted_index=inverted_index)
        tf_idf_dict = prenormalization(tf_dict=tf_dict, idf_array=idf_vector)
        
        start = time.time()
        results, result_nombres, query_vec = tf_idf_knn(
            query_matrix=query_normalized,
            k=8,
            inverted_index=inverted_index,
            tf_idf_dict=tf_idf_dict,
            idf_array=idf_vector,
            centroids=centroids
        )
        times[idx] = time.time() - start
        
        # Limpieza de memoria
        del inverted_index
        del idf_vector
        del tf_dict
        del tf_idf_dict
    
    output_path = os.path.join(os.getcwd(), "src", "Experiments", "times")
    save_timing_data(sizes, times, output_path)
    print("Guardado en ...", output_path)

def save_timing_data(sizes, times, output_path):
    output_file = os.path.join(output_path, 'times_knn_ivf.csv')
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sizes_knn_ivf', 'times_knn_ivf'])
        for size, time_value in zip(sizes, times):
            writer.writerow([size, time_value])

if __name__ == "__main__":
    run_knn_ivf_experiment()