import cupy as cp
import numpy as np
from cuml.cluster import KMeans
import os
import time
from tqdm import tqdm

def normalize_features_gpu(data, batch_size=50000):
    normalized_chunks = []
    n_chunks = (len(data) + batch_size - 1) // batch_size
    print("Normalizando features...")
    
    for i in tqdm(range(0, len(data), batch_size)):
        end_idx = min(i + batch_size, len(data))
        chunk = cp.asarray(data[i:end_idx], dtype=np.float32)
        norms = cp.linalg.norm(chunk, axis=1, keepdims=True)
        chunk_normalized = chunk / cp.maximum(norms, 1e-8)
        normalized_chunks.append(cp.asnumpy(chunk_normalized))
        cp.get_default_memory_pool().free_all_blocks()
    
    return np.concatenate(normalized_chunks)

def generating_visual_words(data, n_clusters=10000, save_path='visual_words.npz', batch_size=50000):
    print(f"Forma del dataset completo: {data.shape}")
    
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    memory_info = cp.cuda.runtime.memGetInfo()
    free_memory = memory_info[0]
    total_memory = memory_info[1]
    print(f"Memoria GPU - Libre: {free_memory/1e9:.2f}GB, Total: {total_memory/1e9:.2f}GB")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        max_iter=100,
        init='k-means||',
        n_init=1,
        output_type='cupy',
        verbose=2,
        batch_size=batch_size
    )
    
    chunks_gpu = []
    n_chunks = (len(data) + batch_size - 1) // batch_size
    
    print(f"\nTransfiriendo datos a GPU en {n_chunks} chunks...")
    for i in tqdm(range(0, len(data), batch_size)):
        end_idx = min(i + batch_size, len(data))
        chunk = cp.asarray(data[i:end_idx], dtype=np.float32)
        chunks_gpu.append(chunk)
    
    print("\nConcatenando datos en GPU...")
    data_gpu = cp.concatenate(chunks_gpu)
    del chunks_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    print("\nEntrenando KMeans...")
    start_time = time.time()
    kmeans.fit(data_gpu)
    
    print("\nGenerando predicciones...")
    visual_words = kmeans.predict(data_gpu)
    
    visual_words_numpy = cp.asnumpy(visual_words)
    centroides = cp.asnumpy(kmeans.cluster_centers_)
    
    del data_gpu, visual_words
    cp.get_default_memory_pool().free_all_blocks()
    
    np.savez(save_path, visual_words=visual_words_numpy, centroides=centroides)
    
    end_time = time.time()
    print(f"\nTiempo total: {end_time - start_time:.2f} segundos")
    return visual_words_numpy






# Uso
loading_path = "/mnt/d/Semestre_2024_2_CS/BD_2/Projects/BD2_Project_3/Proyecto3/data"
file_name = 'features_data_2.npz'

path = os.path.join(loading_path, file_name)





loaded_data = np.load(path, allow_pickle=True)
total_features_index = loaded_data["total_features_index"].item()
total_features = loaded_data["total_features"]



size = 302652
sizes = np.linspace(1000, size, num=10, dtype=int)

for sample_sizes in sizes:

    visual_words_file = f'visual_words_500_v3_{sample_sizes}.npz'
    output_path = os.path.join(loading_path, visual_words_file)
    normalized_features = normalize_features_gpu(total_features, batch_size=50000)
    normalized_features = normalized_features[:sample_sizes*64]

    inicio = time.time()
    visual_words = generating_visual_words(
        normalized_features,
        save_path=output_path,
        n_clusters=500,
        batch_size=50000
    )
    fin = time.time()
    print(f"Tiempo total de procesamiento: {fin - inicio:.2f} segundos")