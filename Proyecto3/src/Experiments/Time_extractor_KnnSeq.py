import os
from tqdm import tqdm
import numpy as np
from Experiments.Experiments_local import process_image_opencv
from Searching.KnnSeq import prepare_knn_model, find_knn_cosine_optimized
import time
import csv

def run_knn_seq_experiment():
    directory_path = os.path.join("F:/", "DB2_Proyect", "portraits")
    path = os.getcwd()
    
    index = 'data/index_dict_color_cv.npz'
    vector = 'data/feature_vector_color_cv.npy'
    
    data_dict = np.load(os.path.join(path,index), allow_pickle=True)
    index_dict = data_dict['arr_0'].item()
    feature_vector = np.load(os.path.join(path,vector))
    
    size = len(feature_vector)
    sizes = np.linspace(1000, size, num=10, dtype=int)
    
    query_path = os.path.join(directory_path,"3504480.jpg")
    _, query = process_image_opencv(query_path)
    query_experiment_index = index_dict["3504480"]
    times = np.zeros(10)
    
    for idx, sample_sizes in tqdm(enumerate(sizes)):
        if sample_sizes < query_experiment_index:
            features = np.concatenate([
                feature_vector[:sample_sizes],
                feature_vector[query_experiment_index:query_experiment_index+1]
            ])
        else:
            features = feature_vector[:sample_sizes]
            
        normalized_features = prepare_knn_model(features)
        start = time.time()
        results = find_knn_cosine_optimized(normalized_features, index_dict, query, directory_path, k=10)
        times[idx] = time.time() - start
    
    def save_timing_data(sizes, times, output_path):
        output_file = os.path.join(output_path, 'times_knn_seq.csv')
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['sizes_knn_seq', 'times_knn_seq'])
            for size, time_value in zip(sizes, times):
                writer.writerow([size, time_value])
    
    output_path = os.path.join(os.getcwd(),"src","Experiments", "times")
    save_timing_data(sizes, times, output_path)
    print("Guardado en ...", output_path)

if __name__ == "__main__":
    run_knn_seq_experiment()