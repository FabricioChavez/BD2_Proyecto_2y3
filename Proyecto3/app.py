from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
from time import time

# project_path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(project_path)

# imports for image querying
from faiss import IndexLSH as lsh
from src.Searching import KnnSeq
from src.Searching import KnnRtree_sql as knnRtree
from src.Searching import RangeSearch
from src.Experiments import Experiments_local as processing
from src.Experiments import Experiments_global as processing_g

# path of imags
portraits_path = os.path.join("D:/", "Documents-D", "datasets", "portraits")

# processed local features 
load_path = os.path.join("D:/", "Documents-D", "datasets", "data", "descriptors_color_2_opencv.npz")
features = np.load(load_path)
# feature vectors
index = 'data/index_dict_color_cv.npz'
vector ='data/feature_vector_color_cv.npy'
path = "D:/Documents-D/datasets/"
data_dict = np.load(os.path.join(path,index), allow_pickle=True)
index_dict = data_dict['arr_0'].item()
feature_vector = np.load(os.path.join(path,vector))
# normalize centroids
normalized_features = KnnSeq.prepare_knn_model(feature_vector)

# lsh
d = len(feature_vector[0])
nbits =1024 # antes 512
lsh_index = lsh(d, nbits)
lsh_index.add(feature_vector)


inverted_dict = {index:name for name , index in index_dict.items()}
def process_results(result , inverted_dict):
    return [os.path.join(portraits_path, inverted_dict[idx] +".jpg" ) for idx in result ]

# variables
results = []
page = 0
image_name = ""
topK = 0
next_page = False
prev_page = False
images_per_page = 9
execution_time = 0

app = Flask(__name__)

def range_query(query, radius):
    print(query, radius)

    # buscar si existe la imagen
    query_path = os.path.join(portraits_path , query)
    print(query_path)
    if not os.path.exists(query_path):
        print("path doesn't exist")
        return []
    
    _, query = processing.process_image_opencv(query_path)

    try:
        results = list( RangeSearch.find_knn_cosine_by_radio(
            normalized_centroids=normalized_features, 
            index_map=index_dict, 
            query_centroid=query, 
            directory_path=portraits_path, 
            radius=radius
        ))
        return results
    except Exception as e:
        print("Failed range query. Error: " + str(e))
        return []


def image_query_knn_sequential(query, k):

    # buscará la imagen en portraits sí o sí
    query_path = os.path.join(portraits_path , query)
    print(query_path)
    if not os.path.exists(query_path):
        # error
        return []
    
    _, query = processing.process_image_opencv(query_path)

    try:
        results = KnnSeq.find_knn_cosine_optimized(
            normalized_centroids=normalized_features,
            index_map=index_dict,
            query_centroid=query,
            directory_path=portraits_path,
            k=k)
        return results
    except Exception as e:
        return []

def image_query_rtree_postgres(query, k):

    # buscará la imagen en portraits sí o sí
    query_path = os.path.join(portraits_path , query)
    print(query_path)
    if not os.path.exists(query_path):
        # error
        return []

    try:
        _, query = processing_g.process_image_global(query_path)
        result = knnRtree.execute_single_query(portraits_path, query, k=k)
        return result
    except Exception as e:
        return []
    
def image_query_lsh(query, k):

    query_path = os.path.join(portraits_path , query)
    if not os.path.exists(query_path):
        # error
        return []
    
    try:
        _, query = processing.process_image_opencv(query_path)

        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = query.astype('float32')
        I, result = lsh_index.search(query, k) 
        results = process_results(result.flatten() , inverted_dict)
        return results
    except Exception as e:
        return []

@app.route('/images/<path:filename>')
def serve_image(filename):
    base_filename = os.path.basename(filename)
    return send_from_directory(portraits_path, base_filename)


@app.route("/", methods=["GET", "POST"])
def search():
    global results
    global page
    global topK
    global image_name
    global next_page
    global prev_page
    global images_per_page
    global execution_time
    # results = []
    t1 = 0
    t2 = 0
    if request.method == "POST":
        image_name = request.form.get("query")  
        topK = request.form.get("topK")  
        source = request.form.get("source")  # Detecta el botón presionado: secuencial, rtree, rango, 

        if not image_name or not topK:
            return render_template("index.html", results=["Error: Consulta o topK vacíos."])

        try:
            # topK = int(topK)
            if source == "load_next_page":
                page += 1
                if len(results) <= images_per_page * ( page + 1):
                    next_page = False
                prev_page = True

            elif source == "load_previous_page":
                page -= 1
                if page == 0:
                    prev_page = False
                next_page = True
            else:
                page = 0
                prev_page = False
                if source == "secuencial":
                    topK = int(topK)
                    t1 = time()
                    results = image_query_knn_sequential(image_name, topK)
                    t2 = time()
                elif source == "rtree":
                    topK = int(topK)
                    t1 = time()
                    results = image_query_rtree_postgres(image_name, topK)
                    t2 = time()
                elif source == "lsh":
                    topK = int(topK)
                    t1 = time()
                    results = image_query_lsh(image_name, topK)
                    t2 = time()
                elif source == "rango":
                    topK= float(topK)
                    t1 = time()
                    results = range_query(image_name, topK)
                    t2 = time()
                    print("Range query #results: ", len(results))
                else:
                    results = []

                execution_time = t2 - t1
                if len(results) > images_per_page:
                    next_page = True
                else:
                    next_page = False
        except Exception as e:
            results = [f"Error: {str(e)}"]
    max_idx = min(len(results), (page + 1) * images_per_page)
    return render_template("index.html", 
                           results=results[page * images_per_page:max_idx], 
                           topK=topK, 
                           image_name=image_name,
                           next_page=next_page,
                           prev_page=prev_page,
                           execution_time = execution_time,
                           page=page)


if __name__ == "__main__":
    app.run(debug=True)
