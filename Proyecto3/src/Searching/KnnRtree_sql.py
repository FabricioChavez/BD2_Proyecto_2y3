import psycopg2
import pandas as pd

# Conexión a la base de datos
conn= psycopg2.connect(
        host="localhost",      # Cambiar si el host es diferente 
        database="",    # cambiar a la base de uso
        user="postgres",       # cambiar al user donde se haran las pruebas 
        password="",  # cambiar al password de uso
        port="5432"
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


def create_table():
        sql_str = f"""
        CREATE SCHEMA IF NOT EXISTS {schema};

        CREATE EXTENSION IF NOT EXISTS cube SCHEMA {schema};

        CREATE TABLE IF NOT EXISTS {schema}.faces (
            id SERIAL,
            image_name VARCHAR(20) NOT NULL,
            vector {schema}.cube, 
            vector_idx {schema}.cube,
            PRIMARY KEY (id, image_name)
        );
        """
        ejecutar_consulta(sql_str)


def set_path():
    sql_str=f"""

    SET search_path to {schema};
    """
    ejecutar_consulta(sql_str)

def delete():
       sql_delete = f"DELETE FROM {schema}.faces;"
       ejecutar_consulta(sql_delete)


import csv
import os
from tqdm import tqdm  # Asegúrate de importar tqdm si lo estás usando

def export_to_csv(features, csv_file):
 
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Escribir encabezado del CSV
        writer.writerow(['image_name', 'vector'])
        for image_name, single_vect in tqdm(features.items(), desc="Exporting features to CSV"):
                # Convertir el vector a formato de texto
                vect_str = ','.join(map(str, single_vect))
                # Escribir la fila en el archivo CSV
                writer.writerow([image_name, vect_str])






def execute_single_query( directory_path, vector , k=5):
    set_path()
    vector_str = ', '.join(map(str, vector))
    sql_str =f"""
            SELECT  id , image_name,
            cube_distance(vector_idx, '({vector_str})') as D
            FROM faces
            ORDER BY vector_idx <-> '({vector_str})'
            LIMIT {k};
        """
    _  , row = ejecutar_consulta(sql_str , select=True)
    return [ os.path.join(directory_path,element +'.jpg') for id , element , sim in row  ]

