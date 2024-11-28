CREATE TABLE IF NOT EXISTS manga(
  title text,
  description text, 
  rating numeric,
  year numeric,
  tags text, 
  cover text, 
  merge text
);

-- se importo data del csv.
-- \copy manga from '</path/to/file/filename.csv>' delimiter ',' CSV HEADER;

-- Prueba y verificacion de los datos
select * from manga;

-- Probando to_tsvector
select to_tsvector('english', merge) as merge_vector from manga;

-- Agregar columna vectorizada a la tabla
alter table manga add column merge_vector tsvector;

-- Actualizar columna
update manga set merge_vector = to_tsvector('english', merge)

-- Probando columna con texto vectorizado
select merge_vector from manga;

-- Agregar un indice invertido (GIN)
create index manga_merge_index on manga using GIN(merge_vector)

explain analyze
select title, merge_vector, ts_rank_cd(merge_vector, query_w) as rank
from manga, to_tsquery('Tanjirou | Kimetsu | nezuko' )
query_w
where query_w @@ merge_vector
order by rank desc limit 10;


----------------------------------------------------------------------

-- Crear la nueva tabla con una columna tsvector
CREATE TABLE manga_vector_subset (
  merge_vector tsvector
);

-- Insertar las primeras 1000 filas desde la tabla manga
INSERT INTO manga_vector_subset (merge_vector)
SELECT merge_vector
FROM manga
LIMIT 1000;

-- Crear el índice GIN en la nueva tabla
CREATE INDEX manga_vector_subset_index
ON manga_vector_subset
USING GIN(merge_vector);

drop table manga_vector_subset

