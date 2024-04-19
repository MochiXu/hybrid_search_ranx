import clickhouse_connect
from numba import NumbaTypeSafetyWarning
from prettytable import PrettyTable
from ranx import Run, fuse
from ranx.normalization import rank_norm
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore', category=NumbaTypeSafetyWarning)
# Use transfromer all-MiniLM-L6-v2
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# MyScale information
host = "10.2.2.71"
port = 8123
username = "default"
password = ""
database = "default"
table = "wiki_abstract_5m"

# Init MyScale client
client = clickhouse_connect.get_client(host=host, port=port,
                                       username=username, password=password)


# Use a table to output your content
def print_results(result_rows, field_names):
    x = PrettyTable()
    x.field_names = field_names
    for row in result_rows:
        x.add_row(row)
    print(x)


terms = "Charted by BGLE"
terms_embedding = model.encode([terms])[0]

vector_search = f"SELECT id, title, body, distance('alpha=3')" \
                f"(body_vector,{list(terms_embedding)}) AS distance FROM {database}.{table} " \
                f"ORDER BY distance ASC LIMIT 10"

vector_search_res = client.query(query=vector_search)
print("\nVectorSearch results:")
print_results(vector_search_res.result_rows, ["ID", "Title", "Body", "Distance"])

text_search = f"SELECT id, title, body, TextSearch(body, '{terms}') AS score " \
              f"FROM {database}.{table} " \
              f"ORDER BY score DESC LIMIT 10"

text_search_res = client.query(query=text_search)
print_results(text_search_res.result_rows, ["ID", "Title", "Body", "Distance"])

# Temporarily store search results.
stored_data = {}
for row in vector_search_res.result_rows:
    stored_data[str(row[0])] = {"title": row[1], "body": row[2]}

for row in text_search_res.result_rows:
    if str(row[0]) not in stored_data:
        stored_data[str(row[0])] = {"title": row[1], "body": row[2]}

# Fusion query results using RRF.
vector_dict = {"query-0": {str(row[0]): float(row[3]) for row in vector_search_res.result_rows}}
bm25_dict = {"query-0": {str(row[0]): float(row[3]) for row in text_search_res.result_rows}}

vector_run = rank_norm(Run(vector_dict, name="vector"))
bm25_run = rank_norm(Run(bm25_dict, name="bm25"))

combined_run = fuse(
    runs=[vector_run, bm25_run],
    method="rrf",
    params={'k': 10}
)


print("\nFusion results:")
pretty_results = []
for id_, score in combined_run.get_doc_ids_and_scores()[0].items():
    if id_ in stored_data:
        pretty_results.append([id_, stored_data[id_]["title"], stored_data[id_]["body"], score])

print_results(pretty_results[:10], ["ID", "Title", "Body", "Score"])

