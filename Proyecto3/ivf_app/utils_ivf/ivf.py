import numpy as np
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def calculate_idf(inverted_index, total_features_index):
    N = len(total_features_index)
    idf_vector = np.zeros(len(inverted_index))
    
    for idx, word_tf_dict in enumerate(inverted_index.values()):
        df = len(word_tf_dict)  
        idf_vector[idx] = np.log10(N / (df + 1))  
    
    return idf_vector

def calculate_tf(inverted_index):
    tf_dict = defaultdict(lambda: np.zeros(len(inverted_index)))
    
    # Calcular TF para cada cluster e imagen
    for cluster_id, cluster_images_tf in tqdm(inverted_index.items()):
        for img_name, tf in cluster_images_tf.items():
            tf_dict[img_name][cluster_id] = np.log1p(tf)
            
    return dict(tf_dict)

def prenormalization(tf_dict, idf_array):
    tf_idf = {}
    for img_name, tf_vector in tqdm(tf_dict.items()):
        tf_idf_vector = tf_vector * idf_array
        tf_idf_norm = np.linalg.norm(tf_idf_vector)
        tf_idf_vector = tf_idf_vector/tf_idf_norm
        tf_idf[img_name] = tf_idf_vector
    return tf_idf

def normalize_query_descriptors(query_matrix):
    """
    Normaliza cada descriptor de 64 dimensiones de la query
    Args:
        query_matrix: Matriz de descriptores donde cada fila es un descriptor de 64 dimensiones
    Returns:
        Matriz normalizada de descriptores
    """
    # Asegurar que sea un array numpy
    query_matrix = np.array(query_matrix)
    
    # Calcular la norma L2 para cada descriptor
    norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
    
    # Evitar división por cero
    norms = np.maximum(norms, 1e-8)
    
    # Normalizar cada descriptor
    query_matrix_normalized = query_matrix / norms
    
    return query_matrix_normalized


def tf_idf_knn(query_matrix, k, inverted_index, tf_idf_dict, idf_array, centroids):
    # Crear y cachear NearestNeighbors
    if not hasattr(tf_idf_knn, 'nbrs'):
        tf_idf_knn.nbrs = NearestNeighbors(
            n_neighbors=1,
            algorithm='kd_tree',
            metric='euclidean',
            leaf_size=30,
            n_jobs=-1
        ).fit(centroids)
    
    # Procesar query en batches
    batch_size = 1000
    indices = []
    for i in range(0, len(query_matrix), batch_size):
        batch = query_matrix[i:i + batch_size]
        _, batch_indices = tf_idf_knn.nbrs.kneighbors(batch)
        indices.append(batch_indices)
    
    indices = np.concatenate(indices).flatten()
    
    # Calcular TF-query usando numpy
    unique_indices, counts = np.unique(indices, return_counts=True)
    tf_query = np.zeros(len(inverted_index))
    tf_query[unique_indices] = np.log1p(counts)
    
    # Normalizar query vector
    tf_idf_query_vector = tf_query * idf_array
    norm = np.linalg.norm(tf_idf_query_vector)
    if norm > 0:
        tf_idf_query_vector_normalized = tf_idf_query_vector/norm
    else:
        return [], set(), tf_idf_query_vector
    
    # Recolectar candidatos relevantes
    relevant_images = set().union(*[inverted_index[idx].keys() for idx in unique_indices])
    print(len(relevant_images))
    # Calcular similitudes en batch
    similarities = [(img_code, np.dot(tf_idf_query_vector_normalized, tf_idf_dict[img_code])) 
                   for img_code in relevant_images]
    
    # Obtener top-k usando sort en lugar de heap
    results = [img_code for img_code, sim in sorted(similarities, key=lambda x: x[1], reverse=True)[:k]]
    
    return results, relevant_images, tf_idf_query_vector_normalized
     
        


