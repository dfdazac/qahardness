import os
import pickle as p
import numpy as np
import matplotlib.pyplot as plt

# Define the base directories
base_dirs = {
    "ConE": "ConE Answers",
    "CQD": "CQD Answers",
    "QTO": "QTO Answers"
}

# Define query structures of interest
query_structures = ["1p", "2p", "3p", "2i", "3i", "ip", "pi"]

# Dictionary to store mappings
query_mappings = {"ConE": {}, "CQD": {}, "QTO": {}}

# Process ConE
cone_path = base_dirs["ConE"]
for dataset in os.listdir(cone_path):
    dataset_path = os.path.join(cone_path, dataset)
    if os.path.isdir(dataset_path):
        for query in query_structures:
            query_file = os.path.join(dataset_path, f"{query}.pkl")
            if os.path.exists(query_file):
                query_mappings["ConE"][query] = query_file

# Process CQD
cqd_path = os.path.join(base_dirs["CQD"], "CQD_FB15k237+H_answers")
if os.path.exists(cqd_path):
    for query in query_structures:
        query_folder = os.path.join(cqd_path, query)
        query_file = os.path.join(query_folder, "rank_dict.pkl")
        if os.path.exists(query_file):
            query_mappings["CQD"][query] = query_file

# Process QTO
qto_path = os.path.join(base_dirs["QTO"], "QTO_FB15k237+H_answers")
if os.path.exists(qto_path):
    for query in query_structures:
        query_file = os.path.join(qto_path, f"rankings_test_{query}.pkl")
        if os.path.exists(query_file):
            query_mappings["QTO"][query] = query_file

# Print the mappings
for method, mapping in query_mappings.items():
    print(f"{method} mappings:")
    for query, file in mapping.items():
        with open(file, "rb") as f:
            data_size = len(p.load(f))
        print(f"  {query}: {file}, size {data_size}")
    print()


def compute_jaccard_similarity(query_mappings, system_A, system_B, query_structure, k):
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
    file_A = query_mappings.get(system_A, {}).get(query_structure)
    file_B = query_mappings.get(system_B, {}).get(query_structure)

    if not file_A or not file_B:
        raise ValueError(f"Missing data for query structure {query_structure} in one of the systems.")

    with open(file_A, 'rb') as f:
        results_A = p.load(f)

    with open(file_B, 'rb') as f:
        results_B = p.load(f)

    jaccard_similarities = []

    for q in results_A:
        ranking_a = results_A[q]
        ranking_b = results_B[q]

        assert len(ranking_a) == len(ranking_b)

        top_k_a = set([i for i, ranking in enumerate(ranking_a) if ranking <= k])
        top_k_b = set([i for i, ranking in enumerate(ranking_b) if ranking <= k])

        len_intersection = len(top_k_a & top_k_b)
        len_union = len(top_k_a | top_k_b)

        if len_union > 0:
            similarity = len_intersection / len_union
            jaccard_similarities.append(similarity)

    return np.mean(jaccard_similarities)

for s in query_structures:
    s_at_k = []
    k_values = (1, 3, 10, 20, 50)
    for k in k_values:
        similarity = compute_jaccard_similarity(query_mappings, 'QTO', 'CQD', s, k)
        print(s, k, f"{similarity:.3f}")
        s_at_k.append(similarity)

    plt.plot(k_values, s_at_k)
    plt.title(f"Overlap@k - {s}")
    plt.xlabel('k')
    plt.ylabel('Jaccard similarity')
    plt.show()