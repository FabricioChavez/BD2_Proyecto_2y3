
SET search_path to project_anime;
--Paso 1 : Pasar la data del .csv a la tabla 
COPY faces (image_name, vector)
FROM 'D:/Semestre_2024_2_CS/BD_2/Projects/Proyecto3/data/features_for_rtree.csv' -- Cambiar a la ruta de donde ejecute el proyecto
WITH (FORMAT csv, HEADER true);

-- Paso 2: Sincronizar la columna `vector_idx` con los valores de `vector`
UPDATE faces
SET vector_idx = vector;


-- Paso 4: Crear el índice GIST para consultas rápidas
-- Incrementar memoria y paralelización
SET maintenance_work_mem = '1.5GB';

SET max_parallel_workers_per_gather = 10;
SET parallel_setup_cost = 0;            -- Reducir el costo de inicialización de procesos paralelos
SET parallel_tuple_cost = 0;            -- Reducir el costo de procesar cada fila en paralelo


-- Crear índice en paralelo
CREATE INDEX CONCURRENTLY vector_idx_gist
ON faces USING GIST (vector_idx);


