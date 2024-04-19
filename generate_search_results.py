import json
import string
from typing import Iterable, Tuple
import clickhouse_connect

import h5py
import tqdm
from clickhouse_connect.driver import Client


def get_query_and_answer(
        file_path: str,
        range_left: int = 10,
        range_right: int = 20
) -> Iterable[Tuple[int, str, list, list, list]]:
    print(f"HDF5 INFO ----- {file_path.split('/')[-1]}, reader range: {range_left}-{range_right}")
    translator = str.maketrans('', '', string.punctuation)
    with h5py.File(file_path, 'r') as train_hdf5:
        query_text = [raw_text.decode('utf-8').translate(translator) for raw_text in
                      train_hdf5['query_text'][range_left:range_right].tolist()]
        query_vector = train_hdf5['test'][range_left:range_right].tolist()
        query_answer = train_hdf5['neighbors'][range_left:range_right].tolist()
        query_distance = train_hdf5['distances'][range_left:range_right].tolist()

        for i in range(range_right - range_left):
            yield range_left + i, query_text[i], query_vector[i], query_answer[i], query_distance[i]


def store_results_in_json(directory_prefix: str, json_dir: str, topk: int, cluster: dict, client: Client):
    qrels_dict = {}
    run_vector_dict = {}
    run_bm25_dict = {}
    for qid, text, vector, answer, distance in tqdm.tqdm(get_query_and_answer(
            f"{directory_prefix}/ms-macro2-768-full-cosine-dev-query.hdf5",
            0,
            6980)):
        vector_search = f"SELECT id, distance('alpha=3')" \
                        f"(vector,{vector}) AS distance FROM {cluster.get('database')}.{cluster.get('table')} " \
                        f"ORDER BY distance ASC LIMIT {topk}"

        text_search = f"SELECT id, TextSearch(text, '{text}') AS score " \
                      f"FROM {cluster.get('database')}.{cluster.get('table')} " \
                      f"ORDER BY score DESC LIMIT {topk}"

        vector_search_res = client.query(query=vector_search)
        text_search_res = client.query(query=text_search)

        qrels_dict[str(qid)] = {str(doc_id): int(score) for doc_id, score in zip(answer, distance)}

        run_bm25_dict[str(qid)] = {str(row[0]): float(row[1]) for row in text_search_res.result_rows}
        run_vector_dict[str(qid)] = {str(row[0]): float(row[1]) for row in vector_search_res.result_rows}

    with open(f"{json_dir}/qrels_ms_macro.json", 'w') as qrels_file:
        json.dump(qrels_dict, qrels_file, indent=2)
    with open(f"{json_dir}/text_search.json", 'w') as run_bm25_file:
        json.dump(run_bm25_dict, run_bm25_file, indent=2)
    with open(f"{json_dir}/vector_search.json", 'w') as run_vector_file:
        json.dump(run_vector_dict, run_vector_file, indent=2)


if __name__ == "__main__":
    # MyScale information
    cluster_info = {
        "host": "10.2.2.71",
        "port": 8123,
        "username": "default",
        "password": "",
        "database": "default",
        "table": "Benchmark",
    }

    # Init MyScale client
    cluster_client: Client = clickhouse_connect.get_client(
        host=cluster_info.get("host"),
        port=cluster_info.get("port"),
        username=cluster_info.get("username"),
        password=cluster_info.get("password")
    )
    dataset_prefix = "/Volumes/970EVO/vector-db-benchmark-datasets/downloaded/ms-macro2-768-full-cosine"
    store_results_in_json(dataset_prefix, "search_results", 100, cluster_info, cluster_client)
