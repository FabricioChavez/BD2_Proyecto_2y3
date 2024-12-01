import numpy as np
import os

def cosine_similarity(vec1, vec2):
    """
    Calcula la similitud de coseno entre dos vectores.

    Parameters:
    vec1 (np.ndarray): Primer vector.
    vec2 (np.ndarray): Segundo vector.

    Returns:
    float: Valor de la similitud de coseno entre vec1 y vec2.
    """
    # Calcula el producto punto de los dos vectores
    dot_product = np.dot(vec1, vec2)
    
    # Calcula la norma (magnitud) de cada vector
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Calcula y retorna la similitud de coseno
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0  # Retorna 0 si alguno de los vectores es cero
    return dot_product / (norm_vec1 * norm_vec2)

def prepare_knn_model(centroids_matrix):
    """
    Prepara el modelo KNN normalizando los centroides.
    
    Parámetros:
    - centroids_matrix (np.ndarray): Matriz donde cada fila es el centroide de una imagen.
    
    Retorna:
    - normalized_centroids (np.ndarray): Matriz de centroides normalizados.
    """
    # Calcular las normas de cada centroid
    norms = np.linalg.norm(centroids_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Evitar división por cero
    
    # Normalizar los centroides
    normalized_centroids = centroids_matrix / norms
    return normalized_centroids

def find_knn_cosine_optimized(normalized_centroids, index_map, query_centroid, directory_path, k=5):
    """
    Encuentra los K vecinos más cercanos utilizando similitud de coseno optimizada.
    
    Parámetros:
    - normalized_centroids (np.ndarray): Matriz de centroides normalizados.
    - index_map (dict): Mapa de nombres de imágenes a índices en normalized_centroids.
    - query_centroid (np.ndarray): Vector de consulta (no normalizado).
    - directory_path (str): Ruta al directorio donde se encuentran las imágenes.
    - k (int): Número de vecinos más cercanos a encontrar.
    
    Retorna:
    - list: Lista de rutas de las K imágenes más cercanas (con ".jpg" añadido).
    """
    # Normalizar el vector de consulta
    query_norm = np.linalg.norm(query_centroid)
    if query_norm == 0:
        raise ValueError("El vector de consulta tiene norma cero.")
    normalized_query = query_centroid / query_norm
    
    # Calcular similitudes de coseno como producto punto
    similarities = normalized_centroids @ normalized_query
    
    # Encontrar los índices de los K mayores valores de similitud
    if k >= len(similarities):
        top_k_indices = np.argsort(-similarities)[:k]
    else:
        top_k_indices = np.argpartition(-similarities, k)[:k]
        # Ordenar los top K índices por similitud
        top_k_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]
    
    # Crear un mapa inverso para obtener nombres a partir de índices
    inv_index_map = {idx: name for name, idx in index_map.items()}
    
    # Obtener los nombres de las imágenes correspondientes y construir las rutas
    knn_results = [os.path.join(directory_path, inv_index_map[idx] + ".jpg") for idx in top_k_indices]
    
    return knn_results

