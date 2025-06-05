import os
import pickle as p
from tqdm import tqdm
import torch
from argparse import ArgumentParser
from collections import defaultdict
import pprint
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


def compute_mrr(scores, query, easy_answers, hard_answers):
    sorted_ids = torch.argsort(scores, descending=True)
    rankings = torch.argsort(sorted_ids)

    easy = list(easy_answers[query])
    hard = list(hard_answers[query])
    num_easy = len(easy)
    num_hard = len(hard)

    # Note order: first easy answers, then hard
    answer_ranks = rankings[easy + hard]
    sorted_ranks, indices = torch.sort(answer_ranks)

    answer_list = torch.arange(num_easy + num_hard, dtype=torch.long)
    filtered_ranks = sorted_ranks - answer_list + 1
    # Recover original order
    filtered_ranks = filtered_ranks[indices.argsort()]
    filtered_ranks = filtered_ranks[num_easy:]

    mrr = torch.mean(1. / filtered_ranks).item()
    return mrr


def main(args):
    query_structures = ["1p", "2p", "3p", "2i", "3i", "ip", "pi"]
    pprint.pprint(vars(args))

    model_name_1 = args.model_1
    model_name_2 = args.model_2
    dataset = args.dataset

    path_1 = os.path.join("answers", model_name_1)
    path_2 = os.path.join("answers", model_name_2)
    scores_path_1 = os.path.join(path_1, dataset)
    scores_path_2 = os.path.join(path_2, dataset)

    print(f"Loading {dataset}...")
    with open(os.path.join('data', dataset, 'test-easy-answers.pkl'), 'rb') as f:
        easy_answers = p.load(f)
    with open(os.path.join('data', dataset, 'test-hard-answers.pkl'), 'rb') as f:
        hard_answers = p.load(f)

    per_structure_mrr_1 = defaultdict(list)
    per_structure_mrr_2 = defaultdict(list)
    per_structure_mrr_combined = defaultdict(list)

    for structure in query_structures:
        scores_file_1 = os.path.join(scores_path_1, structure, f"{model_name_1}_scores.pkl")
        scores_file_2 = os.path.join(scores_path_2, structure, f"{model_name_2}_scores.pkl")

        with open(scores_file_1, "rb") as f_1, open(scores_file_2, "rb") as f_2:
            query_scores_1 = p.load(f_1)

            if args.test:
                query_scores_1_sample = {}
                for i, (k, v) in enumerate(query_scores_1.items()):
                    query_scores_1_sample[k] = v
                    if i == 100:
                        break
                query_scores_1 = query_scores_1_sample

            query_scores_2 = p.load(f_2)

            for query, scores_1 in tqdm(query_scores_1.items(), desc=f"Processing {structure}", mininterval=1.0):
                if len(scores_1) == 1:
                    scores_1 = scores_1[0]

                scores_1 = torch.tensor(scores_1)
                scores_2 = torch.tensor(query_scores_2[query])

                mrr_1 = compute_mrr(scores_1, query, easy_answers, hard_answers)
                mrr_2 = compute_mrr(scores_2, query, easy_answers, hard_answers)
                per_structure_mrr_1[structure].append(mrr_1)
                per_structure_mrr_2[structure].append(mrr_2)
                per_structure_mrr_combined[structure].append(max(mrr_1, mrr_2))

    print("\nMean MRR per structure:")
    for structure in query_structures:
        mean_mrr = torch.tensor(per_structure_mrr_combined[structure]).mean().item()
        print(f"MRR - {structure}: {mean_mrr:.4f}")

    # Perform t-tests and Holm-Bonferroni correction
    p_values = []
    test_labels = []

    for structure in query_structures:
        mrr_1 = torch.tensor(per_structure_mrr_1[structure])
        mrr_2 = torch.tensor(per_structure_mrr_2[structure])
        mrr_combined = torch.tensor(per_structure_mrr_combined[structure])

        # Only run t-test if lengths match (should always be true, but safe check)
        if len(mrr_1) == len(mrr_combined):
            _, p1 = ttest_rel(mrr_1, mrr_combined)
            p_values.append(p1)
            test_labels.append(f"{structure}: model_1 vs combined")

        if len(mrr_2) == len(mrr_combined):
            _, p2 = ttest_rel(mrr_2, mrr_combined)
            p_values.append(p2)
            test_labels.append(f"{structure}: model_2 vs combined")

    # Holm–Bonferroni correction
    reject, corrected_pvals, _, _ = multipletests(p_values, alpha=0.05, method='holm')

    print("\nPaired t-tests with Holm–Bonferroni correction:")
    for i, label in enumerate(test_labels):
        sig = "✅" if reject[i] else "❌"
        print(f"{label}: p = {p_values[i]:.4e}, corrected = {corrected_pvals[i]:.4e} {sig}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_1", choices=["relax", "qto", "cone", "ultra"], default="relax")
    parser.add_argument("--model_2", choices=["relax", "qto", "cone", "ultra"], default="qto")
    parser.add_argument("--dataset", choices=["FB15k237+H"], default="FB15k237+H")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(args)
