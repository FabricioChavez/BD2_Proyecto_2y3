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

#Aquí viene 
    # Ordenar las similitudes y sus índices correspondientes
    sorted_indices = np.argsort(-similarities)  # mayor a menor
    sorted_similarities = similarities[sorted_indices]  # similarities ordenadas


    inv_index_map = {idx: name for name, idx in index_map.items()}

    # for en radios
    for r in radius:
        print(f"Buscando imágenes con radio: {r}")
        # similaridades 
        for i, similarity in zip(sorted_indices, sorted_similarities):
            if similarity >= r:  #si si ya no cumple el radio
            # la ruta
                if i in inv_index_map:
                    image_path = os.path.join(directory_path, inv_index_map[i] + ".jpg")
                    yield image_path  # Yield uno a la vez
                else:
                    raise KeyError(f"Índice {i} no encontrado.")
