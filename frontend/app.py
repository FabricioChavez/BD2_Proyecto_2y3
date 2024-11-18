from flask import Flask, render_template, redirect, request, jsonify
import os
import numpy as np
import pandas as pd
import sys
from query_handler import QueryHandler
import psycopg2
from faiss import IndexLSH as lsh

project_path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
print(project_path)

from Proyecto3.src.Searching import KnnSeq
from Proyecto3.src.Searching import KnnRtree_sql as knnRtree
from Proyecto3.src.Experiments import Experiments_local as processing
from Proyecto3.src.Experiments import Experiments_global as processing_g

app = Flask(__name__)

portraits_path = os.path.join("D:/", "Documents", "datasets", "portraits")

# processed local features 
load_path = os.path.join("D:/", "Documents", "datasets", "data", "descriptors_color_2_opencv.npz")
features = np.load(load_path)
# feature vectors
index = 'data/index_dict_color_cv.npz'
vector ='data/feature_vector_color_cv.npy'
path = "D:/Documents/datasets/"
data_dict = np.load(os.path.join(path,index), allow_pickle=True)
index_dict = data_dict['arr_0'].item()
feature_vector = np.load(os.path.join(path,vector))
# normalize centroids
normalized_features = KnnSeq.prepare_knn_model(feature_vector)

# lsh
d = len(feature_vector[0])
nbits =512
lsh_index = lsh(d, nbits)
lsh_index.add(feature_vector)


inverted_dict = {index:name for name , index in index_dict.items()}
def process_results(result , inverted_dict):
    return [os.path.join(portraits_path, inverted_dict[idx] +".jpg" ) for idx in result ]


# if __name__ == "__main__":
#     app.run(debug=True)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/text-query/own-implementation", methods=["POST"])
def text_query_own_implementation():
    # add regex for query processing?

    # request body
    data = request.get_json()
    query = data["query"]
    k = data["k"]

    # hacer consulta
    path = "../Proyecto2/"
    qh  = QueryHandler(path + "final_index.txt", path + "index_pointer.txt", 70917)
    lista = qh.retrieve_index(query)
    lista = dict(list(lista.items())[:k])

    # cargar sólo ciertas filas del csv, con desfase de 1 por el header
    rows_to_read = list(lista.keys())
    rows_to_read = [0] + [row + 1 for row in rows_to_read]
    df = pd.read_csv(path + "data/data_indexed.csv", 
        usecols=["doc_id", "title", "description", "tags"], 
        skiprows=lambda x: x not in rows_to_read)
    df.set_index("doc_id", inplace = True)

    # df = pd.read_csv(path + "data/data_indexed.csv")
    final_result = []
    # print(df)

    for doc_id, similitude in lista.items():
        manga = {}
        manga["title"] = df.loc[doc_id, "title"]
        manga["description"] = df.loc[doc_id, "description"]
        manga["tags"] = df.loc[doc_id, "tags"]
        manga["rank"] = similitude
        final_result.append(manga)
    return jsonify(final_result)

@app.route("/text-query/postgres", methods=["POST"])
def text_query_postgres():
    # add regex for query processing?

    # request body
    data = request.get_json()
    query = data["query"]
    k = data["k"]

    text = query.lower().strip().replace(" ", " | ")
    sql_tr = f"""
    select title, description, tags, ts_rank_cd(merge_vector, query_w) as rank
    from manga, to_tsquery('{text}' )
    query_w
    where query_w @@ merge_vector
    order by rank desc limit {k};
    """

    conn = psycopg2.connect(
        host="localhost",
        dbname="DBNAME", 
        user="postgres",
        password="PASSWORD",
        port="5432",
        options="-c search_path=SCHEMANAME"
    )

    try:
        cur = conn.cursor()
        result = cur.execute(sql_tr)   
        rows = cur.fetchall()
        print(cur.description)
        column_names = [desc[0] for desc in cur.description]
        result = [dict(zip(column_names, row)) for row in rows]
        return jsonify({
            "statusCode": 200,
            "data": result
        })
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
        return jsonify({
            "message": str(e)
        })
    finally:
        conn.close()

@app.route("/image-query/knn-seq", methods=["POST"])
def image_query_knn_sequential():
    # request body
    data = request.get_json()
    query = data["query"] # nombre de la imagen
    k = data["k"]
    # buscará la imagen en portraits sí o sí
    query_path = os.path.join(portraits_path , query)
    print(query_path)
    if not os.path.exists(query_path):
        return jsonify({
            "statusCode": 404,
            "message": "File not found"
        })
    
    
    _, query = processing.process_image_opencv(query_path)

    try:
        results = KnnSeq.find_knn_cosine_optimized(
            normalized_centroids=normalized_features,
            index_map=index_dict,
            query_centroid=query,
            directory_path=portraits_path,
            k=k)
        return jsonify({
            "statusCode": 200,
            "data": results
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "statusCode": 500,
            "message": str(e)
        })

@app.route("/image-query/rtree", methods=["POST"])
def image_query_rtree_postgres():

    # request body
    data = request.get_json()
    query = data["query"]
    k = data["k"]

    # buscará la imagen en portraits sí o sí
    query_path = os.path.join(portraits_path , query)
    print(query_path)
    if not os.path.exists(query_path):
        return jsonify({
            "statusCode": 404,
            "message": "File not found"
        })

    try:
        _, query = processing_g.process_image_global(query_path)
        result = knnRtree.execute_single_query(portraits_path, query, k=k)
        return jsonify({
            "statusCode":200,
            "data": result
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "statusCode": 500,
            "message": str(e)
        })

@app.route("/image-query/lsh", methods=["POST"])
def image_query_lsh():
    # request body
    data = request.get_json()
    query = data["query"]
    k = data["k"]

    query_path = os.path.join(portraits_path , query)
    print(query_path)
    if not os.path.exists(query_path):
        return jsonify({
            "statusCode": 404,
            "message": "File not found"
        })
    
    try:
        _, query = processing.process_image_opencv(query_path)

        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = query.astype('float32')
        I, result = lsh_index.search(query, k) 
        results = process_results(result.flatten() , inverted_dict)
        return jsonify({
            "statusCode":200,
            "data": results
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "statusCode": 500,
            "message": str(e)
        })

