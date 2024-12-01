from flask import Flask, render_template, request
from Spimi import SPIMIIndex
import pandas as pd
import psycopg2
import time  # Para medir el tiempo

app = Flask(__name__)

# Configuración para el SPIMI
chunk_size = 10000
memory_limit = 500000
index = SPIMIIndex(memory_limit=memory_limit)

try:
    df = pd.read_csv("data.csv")
except Exception as e:
    df = None

# Configuración de PostgreSQL
db_config = {
    'host': 'localhost',
    'dbname': 'p2g6s1',
    'user': 'postgres',
    'password': '1234',
    'port': '5432'
}

@app.route("/", methods=["GET", "POST"])
def search():
    results_df = pd.DataFrame()  
    query_time = None  
    if request.method == "POST":
        query = request.form.get("query")  
        topK = request.form.get("topK")  
        source = request.form.get("source")  

        if not query or not topK:
            return render_template("index.html", results=["Error: Consulta o topK vacíos."])

        try:
            topK = int(topK)
            if df is None:
                raise ValueError("El archivo CSV no se pudo cargar.")

            start_time = time.time()

            if source == "SPIMI":
                query_index = index.retrieve_index(query)
                data = dict(list(query_index.items())[:topK])

                results = []
                for doc_id, score in data.items():
                    merge_value = df.loc[doc_id, "merge"]
                    results.append({"merge": merge_value, "score": score})
                
                results_df = pd.DataFrame(results)

            elif source == "Postgres":
                query_string = " | ".join(query.split())
                with psycopg2.connect(**db_config) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT merge, ts_rank_cd(merge_vector, to_tsquery(%s)) as rank
                            FROM manga
                            WHERE to_tsquery(%s) @@ merge_vector
                            ORDER BY rank DESC LIMIT %s;
                            """, 
                            (query_string, query_string, topK)
                        )
                        results = [{"merge": row[0], "score": row[1]} for row in cur.fetchall()]
                
                results_df = pd.DataFrame(results)

            else:
                raise ValueError("Fuente de búsqueda desconocida.")

            end_time = time.time()
            query_time = round((end_time - start_time) * 1000, 3)

        except Exception as e:
            results = [f"Error: {str(e)}"]

    return render_template("index.html", results_df=results_df.to_dict(orient="records"), query_time=query_time)


if __name__ == "__main__":
    app.run(debug=True)
