import os
import os.path as osp
import pickle as p
import numpy as np
import matplotlib.pyplot as plt

# Define the base directories
methods = {
    "relax"
}

# Define query structures of interest
query_structures = ["1p", "2p", "3p", "2i", "3i", "ip", "pi"]

# Dictionary to store mappings
query_ranks = {m: {} for m in methods}

for method in methods:
    method_path = osp.join('answers', method)
    for dataset in os.listdir(method_path):
        dataset_path = os.path.join(method_path, dataset)
        query_ranks[method][dataset] = {}

        for query in query_structures:
            query_folder = os.path.join(dataset_path, query)
            query_file = os.path.join(query_folder, "query_answer_ranks.pkl")
            if os.path.exists(query_file):
                query_ranks[method][dataset][query] = query_file

# Print the mappings
for method, mapping in query_ranks.items():
    print(f"{method} mappings:")
    for dataset, structure_to_file in mapping.items():
        print(f"\t{dataset}")
        for structure, file in structure_to_file.items():
            with open(file, "rb") as f:
                data_size = len(p.load(f))
            print(f"\t\t{structure}: {file}, {data_size:,} queries")
        print()


def compute_jaccard_similarity(results_A, results_B, k):
    """
    Computes the average Jaccard similarity between the top-k results of two systems for a given query structure.

    Parameters:
    - results_A (dict): A dictionary mapping query structures to the rankings of hard answers for system A
    - results_B (dict): Same, for system B
    - k (int): The ranking threshold to consider.

    Returns:
    - float: The average Jaccard similarity over all query IDs.
    """
    jaccard_similarities = []
    for q in results_A:
        rankings_a = results_A[q]
        rankings_b = results_B[q]

        # Both keys (hard answers) should be identical
        assert rankings_a.keys() == rankings_b.keys()

        top_k_a, top_k_b = [set([e for e, r in rankings.items() if r <= k]) for rankings in (rankings_a, rankings_b)]

        len_intersection = len(top_k_a & top_k_b)
        len_union = len(top_k_a | top_k_b)

        if len_union > 0:
            similarity = len_intersection / len_union
            jaccard_similarities.append(similarity)

    return np.mean(jaccard_similarities)


dataset = "FB15k237+H"
k_values = range(1, 50, 5)
for s in query_structures:
    sim_at_k = []

    file_A = query_ranks["relax"][dataset][s]
    file_B = query_ranks["relax"][dataset][s]

    with open(file_A, 'rb') as f:
        results_A = p.load(f)

    with open(file_B, 'rb') as f:
        results_B = p.load(f)

    for k in k_values:
        similarity = compute_jaccard_similarity(results_A, results_B, k)
        sim_at_k.append(similarity)

    plt.plot(k_values, sim_at_k, label=s)

plt.legend()
plt.title(f"Overlap@k")
plt.xlabel('k')
plt.ylabel('Jaccard similarity')
plt.grid()
plt.show()
