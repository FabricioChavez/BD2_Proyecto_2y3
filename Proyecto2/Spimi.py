import os
import regex as re
import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.data import find
from collections import defaultdict
from sortedcontainers import SortedDict
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import math
import pandas as pd
import heapq
import shelve


class SPIMIIndex:
    def __init__(self, collection, memory_limit=10**6):  # memory limit in bytes
        self.memory_limit = memory_limit
        self.dictionary = None
        self.output_files = []
        self.doc_norms = {}
        self.final_index_file = (
            "final_index.txt"  # Archivo donde se almacenará el índice invertido final
        )
        self.finalPosition = []
        self.collection = collection

    def new_file(self, index):
        filename = f"block_{index}.txt"
        self.output_files.append(filename)
        return filename

    def new_dictionary(self):
        return SortedDict()

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

    def parse_docs(self, documents):
        for doc_id, text in enumerate(documents):
            tokens = self.preprocess_text(text)
            for term in tokens:
                yield term, doc_id

    def add_to_dictionary(self, dictionary, term):
        dictionary[term] = []

    def add_to_postings_list(self, postings_list, doc_id):
        postings_list.append(doc_id)

    def calculate_tf_weights(self, dictionary):
        tf_dict = defaultdict(dict)
        for term, postings_list in dictionary.items():
            for doc_id in postings_list:
                tf = postings_list.count(doc_id)
                tf_dict[term][doc_id] = int(tf)
        return tf_dict

    def write_block_to_disk(self, filename, tf_dict):
        with open(filename, "w", encoding="utf-8") as file:
            for term, postings in tf_dict.items():
                postings_str = ", ".join(
                    f"{doc_id}:{weight}" for doc_id, weight in postings.items()
                )
                file.write(f"{term}: {postings_str}\n")
            final_position = file.tell()
            self.finalPosition.append(final_position)

    def merge_blocks(self):
        # Abriendo todos los bloques para la mezcla
        open_files = [open(f, "r", encoding="utf-8") for f in self.output_files]
        heap = []
        total_blocks = len(self.output_files)
        total_collection = self.collection
        # Inicializar el heap con el primer término de cada bloque
        for i, f in enumerate(open_files):
            line = f.readline().strip()
            if line:
                term, postings = line.split(":", 1)
                postings = postings.strip()
                heapq.heappush(
                    heap, (term, postings, i)
                )  # (term, postings, block file index)
        first_words = [term for term, _, _ in heap]
        last_word = False
        first_word = first_words[0]
        memory_last = 0
        with open(self.final_index_file, "w", encoding="utf-8") as final_index, open(
            "index_pointer.txt", "w", encoding="utf-8"
        ) as index_pointer:
            current_term = None
            current_postings = defaultdict(int)
            final_index.write(f"{total_blocks}, {total_collection}\n")
            current_position = final_index.tell()
            # print("current_position",current_position)
            memory_usage = current_position - 1
            position_pointer = current_position
            while heap:
                term, postings, file_index = heapq.heappop(heap)
                # Parsear los postings de la forma "doc_id: freq"
                term_postings = defaultdict(int)
                for posting in postings.split(", "):
                    doc_id, freq = posting.split(":")
                    doc_id = int(doc_id)
                    freq = int(freq)
                    term_postings[doc_id] += freq  # Sumar frecuencias

                # si el término popeado es diferente del término actual
                if term != current_term:
                    # Escribir el término actual y sus postings al índice final
                    if current_term:
                        postings_str = ", ".join(
                            f"{doc_id}:{freq}"
                            for doc_id, freq in current_postings.items()
                        )
                        line_to_write = f"{current_term}: {postings_str}\n"
                        final_index.write(line_to_write)
                        memory_usage += len(line_to_write)
                        if last_word:
                            if first_word == first_words[0]:
                                memory_last = memory_last - position_pointer
                            index_pointer.write(
                                f"{first_word} {position_pointer} {memory_last}\n"
                            )
                            position_pointer = current_position
                            first_word = current_term
                            last_word = False
                        elif memory_usage > self.memory_limit:
                            last_word = True
                            memory_last = memory_usage
                            memory_usage = 0
                        current_position = final_index.tell()

                    # Actualizar el término actual y sus postings
                    current_term = term
                    current_postings = term_postings
                else:
                    # Si el término popeado es igual al término actual, sumar las frecuencias
                    for doc_id, freq in term_postings.items():
                        current_postings[doc_id] += freq

                # Leer la siguiente línea del bloque correspondiente al término popeado
                next_line = open_files[file_index].readline().strip()
                if next_line:
                    next_term, next_postings = next_line.split(":", 1)
                    next_postings = next_postings.strip()
                    heapq.heappush(heap, (next_term, next_postings, file_index))
            if current_term:
                postings_str = ", ".join(
                    f"{doc_id}:{freq}" for doc_id, freq in current_postings.items()
                )
                final_index.write(f"{current_term}: {postings_str}\n")
            if first_word:
                index_pointer.write(
                    f"{first_word} {position_pointer} {self.memory_limit}\n"
                )
        for f in open_files:
            f.close()
        # eliminar los bloques temporales
        for f in self.output_files:
            os.remove(f)

    def Spimi_invert(self, token_stream):
        self.dictionary = self.new_dictionary()
        output_file = self.new_file(len(self.output_files))
        for term, doc_id in token_stream:
            postings_list = self.dictionary.get(term, [])
            # Escribir el bloque al disco si excede el límite de memoria
            if sys.getsizeof(self.dictionary) > self.memory_limit:
                tf_dict = self.calculate_tf_weights(self.dictionary)
                self.write_block_to_disk(output_file, tf_dict)
                self.dictionary = self.new_dictionary()  # Reset dictionary
                output_file = self.new_file(
                    len(self.output_files)
                )  # Nuevo archivo de salida

            # añadir el término al diccionario si no está presente
            if term not in self.dictionary:
                self.add_to_dictionary(self.dictionary, term)
            postings_list = self.dictionary[term]
            self.add_to_postings_list(postings_list, doc_id)

        # Escribir el último bloque al disco
        if self.dictionary:
            tf_dict = self.calculate_tf_weights(self.dictionary)
            self.write_block_to_disk(output_file, tf_dict)

    def construct_index(self, documents):
        token_stream = self.parse_docs(documents)
        print("Indexing...")
        self.Spimi_invert(token_stream)
        print("Merging blocks...")
        merged_index = self.merge_blocks()
        return merged_index

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

    def binary_search_index_pointer(self, query_term):
        with open("index_pointer.txt", "r", encoding="utf-8") as file:
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

    def specific_block(self, position, size):
        with open(self.final_index_file, "r", encoding="utf-8") as file:
            file.seek(position)
            block = file.read(size)
            return block
