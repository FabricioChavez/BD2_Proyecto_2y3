from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
import sys
import utils_ivf.ivf as ivf
from time import time
import pickle

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import src.Experiments.Experiments_ivf as processing

# SETUP

# Path de imágenes
portraits_directory = os.path.join("D:/", "Documents-D", "datasets", "portraits")

# Cargar archivos
data_path = os.path.join("D:/", "Documents-D", "datasets", "data")
data_name = 'features_data_2.npz'
path = os.path.join(data_path, data_name)
loaded_data = np.load(path, allow_pickle=True)

# Acceder a los datos
total_features_index = loaded_data["total_features_index"].item()  # Convierte a diccionario
total_features = loaded_data["total_features"]  # Es un array de NumPy con todos los features stackeados en forma vertcial , uso solo para obetener centroides

# Achivo de visual words
visual_words_file ='visual_words_500_v2.npz'
vwords_path =os.path.join(data_path,visual_words_file)
loaded_data = np.load(vwords_path, allow_pickle=True)

visual_words = loaded_data["visual_words"]
centroids = loaded_data["centroides"]


def key_generator(num):
    num = num//64
    return tuple((num*64 , num*64+64))


#   Ejecutando todo

# inverted_index = {}

# for idx, cluster_idx in enumerate(visual_words):
#     image_name = total_features_index[key_generator(idx)]
#     if cluster_idx not in inverted_index:
#         inverted_index[cluster_idx] = {}
    
#     if image_name not in inverted_index[cluster_idx]:
#         inverted_index[cluster_idx][image_name] = 1
#     else:
#         inverted_index[cluster_idx][image_name] += 1


# # idf, tf, tfidf
# idf_inverted_index = ivf.calculate_idf(inverted_index=inverted_index , 
#                                        total_features_index=total_features_index)    
# tf_dict = ivf.calculate_tf(inverted_index)
# tf_idf_dict = ivf.prenormalization(tf_dict=tf_dict, idf_array=idf_inverted_index)

#    O habiendo guardado todo en archivos

with open("resources/inverted_index.pkl", "rb") as file:
    inverted_index = pickle.load(file)

idf_inverted_index = np.load("resources/idf_inverted_index.npy")

with open("resources/tf_dict.pkl", "rb") as file:
    tf_dict = pickle.load(file)

with open("resources/tf_idf_dict.pkl", "rb") as file:
    tf_idf_dict = pickle.load(file)


def process_query(image_name, k):
    query_path = os.path.join(portraits_directory, image_name)
    if not os.path.exists(query_path):
        print("path doesn't exist")
        return []
    
    try:
        _,query= processing.process_image_opencv(query_path)
        query_normalized = ivf.normalize_query_descriptors(query_matrix=query)
        results, _, _ = ivf.tf_idf_knn(
            query_matrix=query_normalized,
            k=k,
            inverted_index=inverted_index,
            tf_idf_dict=tf_idf_dict,
            idf_array=idf_inverted_index,
            centroids=centroids
        )
        results = [ img_name +'.jpg' for img_name in results] # [3504480.jpg, ...]

        return results

    except Exception as e:
        print(f"Error: {str(e)}")
        return []
    
print("Finished setup. -----------")

results = []
page = 0
top_k = 0
next_page = False
prev_page = False
images_per_page = 9
execution_time = 0
image_name= ""

app = Flask(__name__)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(portraits_directory, filename)

@app.route("/", methods=["GET", "POST"])
def search():
    global results
    global page
    global top_k
    global image_name
    global next_page
    global prev_page
    global images_per_page
    global execution_time
    t1 = 0 # medir tiempos
    t2 = 0

    if request.method == "POST":

        image_name = request.form.get("query")
        top_k = request.form.get("topK")  
        source = request.form.get("source")

        if not image_name or not top_k:
            return render_template("index.html", results=["Error: Consulta o topK vacíos."])
        
        try:
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
            else: # source == "search"
                page = 0
                prev_page = False
                top_k = int(top_k)
                t1 = time()
                results = process_query(image_name=image_name, k=top_k)
                t2 = time()

                execution_time = t2 - t1
                if len(results) > images_per_page:
                    next_page = True
                else:
                    next_page = False
        except Exception as e:
            print(f'Error: {str(e)}')
            results = []

    max_idx = min(len(results), (page + 1) * images_per_page) # hasta qué parte del array renderizamos?
    return render_template("index.html",
                           results=results[page * images_per_page:max_idx],
                           topK=top_k,
                           image_name=image_name,
                           next_page=next_page,
                           prev_page=prev_page,
                           execution_time=execution_time,
                           page=page)

if __name__ == "__main__":
    app.run(debug=True)
