import os
import pickle as p
from tqdm import tqdm

query_structures = ["1p", "2p", "3p", "2i", "3i", "ip", "pi"]

# Process ConE
cone_path = "ConE Answers"
for dataset in os.listdir(cone_path):
    if not dataset.startswith("ConE"):
        continue

    dataset_path = os.path.join(cone_path, dataset)

    dataset_name = dataset[5:-8]
    print(f"Processing {dataset_name}")

    with open(os.path.join('data', dataset_name, 'test-easy-answers.pkl'), 'rb') as f:
        easy_answers = p.load(f)

    with open(os.path.join('data', dataset_name, 'test-hard-answers.pkl'), 'rb') as f:
        hard_answers = p.load(f)

    for structure in tqdm(query_structures):
        query_file = os.path.join(dataset_path, f"{structure}.pkl")
        with open(query_file, "rb") as f:
            query_data = p.load(f)

        for query, rankings in query_data.items():
            easy = easy_answers[query]
            hard = hard_answers[query]
            assert len(easy) + len(hard) == len(rankings)
            # Get rid of rankings for easy answers
            query_data[query] = rankings[len(easy):]

        with open(os.path.join(dataset_path, f"{structure}-hard-rankings.pkl"), "wb") as f:
            p.dump(query_data, f)
