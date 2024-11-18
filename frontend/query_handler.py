from collections import Counter, defaultdict
import math
import nltk
import os
import pandas as pd
import regex as re
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.data import find
from nltk.corpus import stopwords

class QueryHandler:
    
    def __init__(self, index_file, indexpointer_file, collection):
        self.index_file  = index_file
        self.indexpointer_file = indexpointer_file
        self.collection = collection

    def specific_block(self, position, size):
        with open(self.index_file, "r", encoding="utf-8") as file:
            file.seek(position)
            block = file.read(size)
            return block
    
    def binary_search_index_pointer(self, query_term):
        with open(self.indexpointer_file, "r", encoding="utf-8") as file:
            pointer_data = file.readlines()
        left, right = 0, len(pointer_data) - 1
        mid = (left + right) // 2
        while left <= right:
            mid = (left + right) // 2
            line = pointer_data[mid].strip().split()
            term = line[0]
            if term == query_term:
                return int(line[1]), int(
                    line[2]
                )  # Retornar posición y tamaño del bloque
            elif term < query_term:
                if mid + 1 < len(pointer_data):
                    next_term = pointer_data[mid + 1].strip().split()[0]
                    if term < query_term < next_term:
                        return int(line[1]), int(line[2])
                left = mid + 1
            else:
                right = mid - 1
        # retornar el ultimo bloque
        return int(line[1]), int(line[2])
    
    def parse_block(self, block):
        data_block = defaultdict(dict)
        lines = block.split("\n")
        # quitar valores na
        lines = list(filter(None, lines))
        for line in lines:
            # print("line",line)
            term, postings = line.split(":", 1)
            postings = postings.strip()
            postings = postings.split(", ")
            postings = {
                int(doc_id): int(freq)
                for doc_id, freq in (posting.split(":") for posting in postings)
            }
            data_block[term] = postings
        return data_block
    
    def retrieve_index(self, query):
        # preprocesar la consulta
        query = self.preprocess_text(query)
        query_tf_idf = {}
        score = {}
        query_tf = Counter(query)
        query_tf_idf = {}
        query_length = 0
        doc_lengths = {}
        print("Query:", query)
        for term in query:
            block = self.binary_search_index_pointer(term)
            data_block = self.specific_block(block[0], block[1])
            data_block = self.parse_block(data_block)
            # calcular tf-idf
            idf = math.log10(self.collection / len(data_block[term]))
            query_tf_idf[term] = math.log10(1 + query_tf[term]) * idf
            query_length += query_tf_idf[term] ** 2

            for doc_id, freq in data_block[term].items():
                tf = math.log10(1 + freq)
                tf_idf = tf * idf
                score[doc_id] = score.get(doc_id, 0) + tf_idf * query_tf_idf[term]
                if doc_id not in doc_lengths:
                    doc_lengths[doc_id] = 0
                doc_lengths[doc_id] += tf_idf**2

        query_length = math.sqrt(query_length)
        # calculando la similitud de coseno
        for doc_id in score:
            score[doc_id] /= query_length * math.sqrt(doc_lengths[doc_id])

        # sortear por puntaje
        score = dict(sorted(score.items(), key=lambda item: item[1], reverse=True))
        return score
    def preprocess_text(self, text):
        current_dir = os.getcwd()
        if current_dir not in nltk.data.path:
            nltk.data.path.append(current_dir)

        for resource in ["punkt", "punkt_tab"]:
            try:
                find(f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, download_dir=current_dir, quiet=True)

        text = re.sub(r"[^\p{L}\s]", "", text)  # Remove special characters
        tokens = word_tokenize(text.lower(), language="english")
        stop_words = set(stopwords.words("english"))  # Stopwords in English
        tokens_filtrados = [word for word in tokens if word not in stop_words]
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens_filtrados]
        return tokens
    
# def handle_query(query):
#     qh  = QueryHandler("final_index.txt", "index_pointer.txt", 70917)
#     lista = qh.retrieve_index(query)
#     lista = dict(list(lista.items())[:5])
#     df = pd.read_csv("data/data_final2.csv")
#     final_result = []

#     for doc_id in lista:
#         # print(df.loc[doc_id, "merge"])
#         texts = df.loc[doc_id, "merge"]
#         final_result.append(texts)
#         # print("\n")
#     return final_result
# if __name__ == "__main__":
#     qh  = QueryHandler("final_index.txt", "index_pointer.txt", 70917)
#     query = "Yuki college student"

#     lista = qh.retrieve_index(query)
#     print(lista)
#     lista = dict(list(lista.items())[:5])
#     final_result = []
#     # print(lista)
#     df = pd.read_csv("data/data_final2.csv")
#     for doc_id in lista:
#         # print(df.loc[doc_id, "merge"])
#         final_result.append(texts)
#         texts = df.loc[doc_id, "merge"]
#         # print("\n")
    