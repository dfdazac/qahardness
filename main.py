import os
import pickle as p
import numpy as np
import matplotlib.pyplot as plt

# Define the base directories
base_dirs = {
    "ConE": "ConE Answers",
    "CQD": "CQD Answers",
    "QTO": "QTO Answers",
    "CR": "CR Answers"
}

# Define query structures of interest
query_structures = ["1p", "2p", "3p", "2i", "3i", "ip", "pi"]

# Dictionary to store mappings
query_mappings = {"ConE": {}, "CQD": {}, "QTO": {}, "CR": {}}

# Process ConE
cone_path = base_dirs["ConE"]
for dataset in os.listdir(cone_path):
    dataset_name = dataset[5:-8]
    dataset_path = os.path.join(cone_path, dataset)
    query_mappings["ConE"][dataset_name] = {}
    if os.path.isdir(dataset_path):
        for query in query_structures:
            query_file = os.path.join(dataset_path, f"{query}-hard-rankings.pkl")
            if os.path.exists(query_file):
                query_mappings["ConE"][dataset_name][query] = query_file

# Process CQD and CR
for method in ("CQD", "CR"):
    path = base_dirs[method]
    for dataset in os.listdir(path):
        dataset_name = dataset[len(method) + 1:-8]
        dataset_path = os.path.join(path, dataset)
        query_mappings[method][dataset_name] = {}
        if os.path.exists(path):
            for query in query_structures:
                query_folder = os.path.join(dataset_path, query)
                query_file = os.path.join(query_folder, "rank_dict.pkl")
                if os.path.exists(query_file):
                    query_mappings[method][dataset_name][query] = query_file

# Process QTO
qto_path = base_dirs["QTO"]
for dataset in os.listdir(qto_path):
    dataset_name = dataset[4:-8]
    dataset_path = os.path.join(qto_path, dataset)
    query_mappings["QTO"][dataset_name] = {}
    if os.path.exists(qto_path):
        for query in query_structures:
            query_file = os.path.join(dataset_path, f"rankings_test_{query}.pkl")
            if os.path.exists(query_file):
                query_mappings["QTO"][dataset_name][query] = query_file

# Print the mappings
for method, mapping in query_mappings.items():
    print(f"{method} mappings:")
    for dataset, structure_to_file in mapping.items():
        print(f"\t{dataset}")
        for structure, file in structure_to_file.items():
            with open(file, "rb") as f:
                data_size = len(p.load(f))
            print(f"\t\t{structure}: {file}, {data_size:,} queries")
        print()


def compute_jaccard_similarity(query_mappings, system_A, system_B, query_structure, k, dataset):
    """
    Computes the average Jaccard similarity between the top-k results of two systems for a given query structure.

    Parameters:
    - query_mappings (dict): Dictionary mapping query structures to pickle file paths for each system.
    - system_A (str): Name of the first system (e.g., 'ConE').
    - system_B (str): Name of the second system (e.g., 'QTO').
    - query_structure (str): Query structure to compare (e.g., '1p').
    - k (int): The ranking threshold to consider.

    Returns:
    - float: The average Jaccard similarity over all query IDs.
    """
    # Load results for system A
    file_A = query_mappings[system_A][dataset][query_structure]
    file_B = query_mappings[system_B][dataset][query_structure]

    with open(file_A, 'rb') as f:
        results_A = p.load(f)

    with open(file_B, 'rb') as f:
        results_B = p.load(f)

    jaccard_similarities = []
    # print(query_structure)
    # print(f"\t{system_A} query: {next(iter(results_A))}")
    # print(f"\t{system_B} query: {next(iter(results_B))}")

    for q in results_A:
        if q in results_A and q in results_B:
            ranking_a = results_A[q]
            ranking_b = results_B[q]
        else:
            continue

        assert len(ranking_a) == len(ranking_b)

        top_k_a = set([i for i, ranking in enumerate(ranking_a) if ranking <= k])
        top_k_b = set([i for i, ranking in enumerate(ranking_b) if ranking <= k])

        len_intersection = len(top_k_a & top_k_b)
        len_union = len(top_k_a | top_k_b)

        if len_union > 0:
            similarity = len_intersection / len_union
            jaccard_similarities.append(similarity)

    return np.mean(jaccard_similarities)

k_values = range(1, 50, 5)
for s in query_structures:
    sim_at_k = []
    for k in k_values:
        similarity = compute_jaccard_similarity(query_mappings, 'QTO', 'CR', s, k, "FB15k237+H")
        print(s, k, f"{similarity:.3f}")
        sim_at_k.append(similarity)

    plt.plot(k_values, sim_at_k, label=s)

plt.legend()
plt.title(f"Overlap@k")
plt.xlabel('k')
plt.ylabel('Jaccard similarity')
plt.grid()
plt.savefig("overlap@k.pdf")
