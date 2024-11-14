from rtree import index
import numpy as np
from collections import defaultdict
from tqdm import tqdm

class RtreeSupportRAM:
    def __init__(self):
       
        # Initialize properties for a 16-dimensional R-tree
        p = index.Property()
        p.dimension = 20

        # Create an in-memory index
        print("Creating an in-memory R-tree index...")
        self.idx_ram = index.Index(properties=p)
        print("In-memory index created.")

        self.insertions_count = 0

    def insert(self, features):
        
        print("Starting insertion of features into the in-memory index...")
        for img_name, vector in tqdm(features.items(), desc="Inserting features"):
         
                # Ensure the vector is a list of floats
                vector = [float(coord) for coord in vector]

                vector_id = img_name
                unique_id = hash(vector_id)

                bbox = tuple(vector) + tuple(vector)

                
                self.idx_ram.insert(id=unique_id, coordinates=bbox, obj=vector_id)

                self.insertions_count += 1

        print(f"Inserted {self.insertions_count} vectors into the in-memory index.")

    def KNN_LOCAL(self, vector, k):
        
        # Ensure the vector is a list of floats
        vector = [float(coord) for coord in vector]

        # Create a bounding box for the query vector (min and max are the same)
        bbox = tuple(vector) + tuple(vector)

        # Query the in-memory index for k nearest neighbors
        results = list(self.idx_ram.nearest(coordinates=bbox, num_results=k, objects=True))

        # Extract the identifiers of the nearest neighbors
        vector_ids = [res.object + '.jpg' for res in results]
        return vector_ids

    def KNN_GLOBAL(self, query_matrix, k):
       
        image_votes = defaultdict(int)
        print("Starting global KNN query...")
        for vector in tqdm(query_matrix, desc="Processing local descriptors"):
            # Find k nearest neighbors for each local vector
            vector_k_neighbors = self.KNN_LOCAL(vector, k)
            for neighbor in vector_k_neighbors:
                img_name = neighbor[0]  # Extract image name from (img_name, idx)
                image_votes[img_name] += 1  # Increment vote for the image

        # Sort images by the number of votes in descending order
        sorted_images = sorted(image_votes.items(), key=lambda x: x[1], reverse=True)
        # Return the top k images
        top_k_images = [img for img, votes in sorted_images[:k]]
        return top_k_images
