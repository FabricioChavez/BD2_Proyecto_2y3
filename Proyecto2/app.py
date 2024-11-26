from flask import Flask, render_template, request
from Spimi import SPIMIIndex
import pandas as pd
import psycopg2

app = Flask(__name__)

# Configuración
chunk_size = 10000
memory_limit = 500000
index = SPIMIIndex(memory_limit=memory_limit)

# Cargar el archivo CSV
try:
    df = pd.read_csv("data.csv")
except Exception as e:
    df = None

# Configuración de PostgreSQL
db_config = {
    'dbname': 'dbname',
    'user': 'postgres',
    'password': 'password',
    'host': 'localhost',
    'port': '5432'
}

@app.route("/", methods=["GET", "POST"])
def search():
    results = []
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

            if source == "SPIMI":
                query_index = index.retrieve_index(query)
                results = [df.loc[doc_id, "merge"] for doc_id in list(query_index.keys())[:topK]]
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
                        results = [row[0] for row in cur.fetchall()]
            else:
                raise ValueError("Fuente de búsqueda desconocida.")
        except Exception as e:
            results = [f"Error: {str(e)}"]

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
