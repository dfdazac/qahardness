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

with open(os.path.join('data', 'FB15k-237+H', 'test-easy-answers.pkl'), 'rb') as f:
    easy_answers = p.load(f)

with open(os.path.join('data', 'FB15k-237+H', 'test-hard-answers.pkl'), 'rb') as f:
    hard_answers = p.load(f)

# Process ConE
cone_path = base_dirs["ConE"]
for dataset in os.listdir(cone_path):
    dataset_path = os.path.join(cone_path, dataset)
    if os.path.isdir(dataset_path):
        for query in query_structures:
            query_file = os.path.join(dataset_path, f"{query}.pkl")
            with open(query_file, "rb") as f:
                query_data = p.load(f)
            assert next(iter(query_data)) in easy_answers
            assert next(iter(query_data)) in hard_answers