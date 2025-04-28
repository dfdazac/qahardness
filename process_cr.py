import os
import pickle as p
from tqdm import tqdm
import torch

query_structures = ["1p", "2p", "3p", "2i", "3i", "ip", "pi"]

cr_path = "CR Answers"
for dataset in os.listdir(cr_path):
    if not dataset.startswith("CR"):
        continue

    dataset_path = os.path.join(cr_path, dataset)

    dataset_name = dataset[3:-8]
    print(f"Processing {dataset_name}")

    with open(os.path.join('data', dataset_name, 'test-easy-answers.pkl'), 'rb') as f:
        easy_answers = p.load(f)

    with open(os.path.join('data', dataset_name, 'test-hard-answers.pkl'), 'rb') as f:
        hard_answers = p.load(f)

    for structure in tqdm(query_structures):
        query_file = os.path.join(dataset_path, structure, "CombinedRanker_scores.pkl")

        with open(query_file, "rb") as f:
            query_data = p.load(f)

        for query, order in query_data.items():
            easy = easy_answers[query]
            hard = hard_answers[query]

            rankings = torch.argsort(torch.tensor(order))
            # rankings[i] = ranking of entity i

            cur_ranking = rankings[list(hard)]
            # Apply filtering
            cur_ranking, indices = torch.sort(cur_ranking)
            answer_list = torch.arange(len(hard), dtype=torch.float)
            cur_ranking = cur_ranking - answer_list + 1
            # Recover original order
            cur_ranking = cur_ranking[indices.argsort()]

            # Get rid of rankings for easy answers
            query_data[query] = cur_ranking.int().tolist()

        with open(os.path.join(dataset_path, structure, "rank_dict.pkl"), "wb") as f:
            p.dump(query_data, f)
