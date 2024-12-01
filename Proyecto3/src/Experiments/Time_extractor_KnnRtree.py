import os
from tqdm import tqdm
import numpy as np
import time
import csv
import pandas as pd
import psycopg2
import time
from Experiments_global import process_image_global

# Conexión a la base de datos
conn = psycopg2.connect(
    host="localhost",  # Cambiar si el host es diferente
    database="postgres",  # cambiar a la base de uso
    user="postgres",  # cambiar al user donde se haran las pruebas
    password="123",  # cambiar al password de uso
    port="5432",
)

schema = "project_anime"

try:

    cursor = conn.cursor()

    cursor.execute("SELECT version();")

    version = cursor.fetchone()
    print(f"Versión de PostgreSQL: {version}")

except Exception as error:
    print(f"Error al conectar con PostgreSQL: {error}")


def ejecutar_consulta(sql_str, select=False):
    try:
        # Crear un cursor para ejecutar las consultas
        cur = conn.cursor()
        # Ejecutar la consulta
        cur.execute(sql_str)
        # Aplicar commit si la consulta es INSERT, UPDATE o DELETE
        conn.commit()

        # Manejo para consultas SELECT
        if select:
            rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
            return df, rows
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()  # Revertir si hay un error
    finally:
        cur.close()

### scripts SQL

def recreate_table():
    """
    Recrea la tabla 'faces' para asegurar que esté vacía antes de insertar nuevos datos.
    """
    sql_str = f"""
    CREATE SCHEMA IF NOT EXISTS {schema};
    CREATE EXTENSION IF NOT EXISTS cube SCHEMA {schema};
    DROP TABLE IF EXISTS {schema}.faces;
    CREATE TABLE {schema}.faces (
        id SERIAL,
        image_name VARCHAR(20) NOT NULL,
        vector {schema}.cube,
        vector_idx {schema}.cube,
        PRIMARY KEY (id, image_name)
    );
    """
    ejecutar_consulta(sql_str)


def set_path():
    sql_str = f"""

    SET search_path to {schema};
    """
    ejecutar_consulta(sql_str)


def save_subset_data(data, path):
    data.to_csv(path, index=False)


def execute_single_query(vector, k=5):
    set_path()
    vector_str = ", ".join(map(str, vector))
    sql_str = f"""
            SELECT  id , image_name,
            cube_distance(vector_idx, '({vector_str})') as D
            FROM faces
            ORDER BY vector_idx <-> '({vector_str})'
            LIMIT {k};
        """
    _, row = ejecutar_consulta(sql_str, select=True)
    return [element for id, element, sim in row]
    # return [os.path.join(directory_path, element + ".jpg") for id, element, sim in row]

def copy_data_to_postgres(csv_path):
    """
    Carga los datos desde un archivo CSV a PostgreSQL usando `COPY`.
    """
    sql_str = f"""
    COPY {schema}.faces (image_name, vector)
    FROM '{csv_path}'
    WITH (FORMAT csv, HEADER true);
    """
    ejecutar_consulta(sql_str)


def synchronize_vector_idx():
    """
    Sincroniza la columna `vector_idx` con `vector`.
    """
    sql_str = f"""
    UPDATE {schema}.faces
    SET vector_idx = vector;
    """
    ejecutar_consulta(sql_str)


def create_gist_index():
    """
    Crea el índice GIST para consultas rápidas sin transacción explícita.
    """
    try:
        conn.autocommit = True  # Desactivar transacciones automáticas
        sql_str = f"""
        CREATE INDEX CONCURRENTLY vector_idx_gist
        ON {schema}.faces USING GIST (vector_idx);
        """
        ejecutar_consulta(sql_str)
    finally:
        conn.autocommit = False  # Restaurar comportamiento por defecto


def measure_execution_time(vector, k=5):
    """
    Mide el tiempo de ejecución de una consulta fija.
    """
    start_time = time.time()
    execute_single_query(vector, k)  # Consulta fija
    elapsed_time = time.time() - start_time
    return elapsed_time


def run_knn_Rtree_experiment():
    directory_path = os.path.join("F:/", "DB2_Proyect", "portraits")    # poner la dirección de la carpeta de imágenes

    # poner la ruta de la carpeta de features
    data = pd.read_csv("features_for_rtree.csv")

    size = len(data)
    sizes = np.linspace(1000, size, num=10, dtype=int)

    # Consulta fija
    query_path = os.path.join(directory_path, "10003410.jpg")
    _, query = process_image_global(query_path)

    # Resultados de tiempos
    results = []

    # Ruta temporal para los CSVs
    temp_csv_path = os.path.join(os.getcwd(), "temp_subset.csv")

    for size in sizes:
        subset = data[:size]
        recreate_table()  # Vaciar y recrear la tabla
        save_subset_data(subset, temp_csv_path)  # Guardar el subconjunto
        copy_data_to_postgres(temp_csv_path)  # Cargar los datos

        # Sincronizar `vector_idx` y crear índice
        synchronize_vector_idx()
        create_gist_index()
        # Medir el tiempo de ejecución de la consulta fija
        elapsed_time = measure_execution_time(query)
        results.append((size, elapsed_time))
        print(f"Tamaño: {size}, Tiempo: {elapsed_time:.8f} segundos")

    # Eliminar el archivo CSV temporal
    os.remove(temp_csv_path)
    # Guardar resultados en un archivo CSV
    results_df = pd.DataFrame(results, columns=["sizes_knn_rtree", "times_knn_rtree"])
    results_df.to_csv("time_knn_rtree.csv", index=False)
    print("Resultados guardados en 'time_knn_rtree.csv'")

    conn.close()

if __name__ == "__main__":
    run_knn_Rtree_experiment()
