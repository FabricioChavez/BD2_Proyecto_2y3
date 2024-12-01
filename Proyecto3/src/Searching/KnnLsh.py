import cv2 
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tqdm
import sklearn  
from faiss import IndexLSH as lsh
import os
from skimage import color , io

class LSH_wrapper:
    def __init__ (self  , dimension = 0 , nbits = 512, directory_path="" ):
        self.dimension = dimension
        self.nbits = nbits
        self.index = lsh(self.dimension , self.nbits)
        self.directory_path = directory_path

    def process_results(self ,  result , inverted_dict):
        return [os.path.join(self.directory_path, inverted_dict[idx] +".jpg" ) for idx in result ]   
     
    def Indexing(self , feature_vector):
        self.index.add(feature_vector)

    def Knn_lsh(self , query , k =8 , inverted_dict = {}):
        I , result  = self.index.search(query , k)
        return self.process_results(result.flatten() , inverted_dict)
       