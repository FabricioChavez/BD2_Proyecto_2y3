import numpy as np
import os

def find_knn_cosine_by_radio(normalized_centroids, index_map, query_centroid, directory_path, radius):
    """
    Encuentra los vecinos dentro de un radio utilizando similitud de coseno optimizada.
    Retorna un generador que produce las rutas de imágenes de los vecinos encontrados.

    Parámetros:
    - normalized_centroids (np.ndarray): Matriz de centroides normalizados.
    - index_map (dict): Mapa de nombres de imágenes a índices en normalized_centroids.
    - query_centroid (np.ndarray): Vector de consulta (no normalizado).
    - directory_path (str): Ruta al directorio donde se encuentran las imágenes.
    - radius (float): Umbral de similitud de coseno (radio) para incluir vecinos.

    Retorna:
    - generator: Generador que produce rutas de imágenes dentro del radio.
    """
    # Normalizar el vector de consulta
    query_norm = np.linalg.norm(query_centroid)
    if query_norm == 0:
        raise ValueError("El vector de consulta tiene norma cero.")
    normalized_query = query_centroid / query_norm

    # Calcular similitudes de coseno como producto punto
    similarities = normalized_centroids @ normalized_query


#Aquí vienen las variaciones para el rango

    #mapa inverso para obtener nombres a partir de índices
    inv_index_map = {idx: name for name, idx in index_map.items()}

    # for sobre las similitudes y filtrar por radio
    # results = []
    for idx, similarity in enumerate(similarities):
        if similarity <= radius:
            # ruta con indice 
            image_path = os.path.join(directory_path, inv_index_map[idx] + ".jpg")
            yield image_path  # yield uno a la vez
            # results.append(image_path)
            
