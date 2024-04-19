import json
from typing import Tuple, List

from numba import NumbaTypeSafetyWarning
from ranx import Run, Qrels, compare, fuse, optimize_fusion
from ranx.normalization import min_max_norm
from min_max_inverted import min_max_norm_inverted

import warnings

# 忽略 NumbaTypeSafetyWarning: unsafe cast from uint64 to int64. Precision may be lost.
warnings.filterwarnings('ignore', category=NumbaTypeSafetyWarning)


def load_search_results(json_dir: str) -> Tuple[dict, dict, dict]:
    with open(f"{json_dir}/qrels_ms_macro.json", 'r') as qrels_file:
        qrels_dict = json.load(qrels_file)
    with open(f"{json_dir}/text_search.json", 'r') as run_bm25_file:
        run_bm25_dict = json.load(run_bm25_file)
    with open(f"{json_dir}/vector_search.json", 'r') as run_vector_file:
        run_vector_dict = json.load(run_vector_file)
    return qrels_dict, run_bm25_dict, run_vector_dict


def compare_search_results(qrels: Qrels, runs: List[Run]):
    report = compare(
        qrels=qrels,
        runs=runs,
        # metrics=["map@100", "mrr@100", "ndcg@100", "map@100"],
        metrics=["mrr@10", "map@10", "ndcg@10"],
        max_p=0.01,  # P-value threshold
    )
    print(report)


def search_results_fusion(qrels: Qrels, runs: List[Run], fusion_methods: List[str]):
    combined_runs = []
    for fusion_method in fusion_methods:
        best_params = optimize_fusion(
            qrels=qrels,
            runs=runs,
            method=fusion_method,
            metric="mrr@10",  # The metric to maximize during optimization
            return_optimization_report=False
        )
        print(f"Best params for {fusion_method} is: {best_params}")
        combined_run = fuse(
            runs=runs,
            method=fusion_method,
            params=best_params
        )
        combined_runs.append(combined_run)
    compare_search_results(qrels, runs + combined_runs)


if __name__ == '__main__':
    fusion_methods_with_optimize = ['rrf', 'gmnz', 'probfuse', 'slidefuse', 'bayesfuse', 'wmnz', 'rbc', 'logn_isr',
                                    'posfuse', 'segfuse', 'mapfuse', 'w_bordafuse', 'w_condorcet', 'mixed', 'wsum']
    qrels_dict, bm25_dict, vector_dict = load_search_results("search_results")
    qrels = Qrels(qrels_dict)
    bm25_run = min_max_norm(Run(bm25_dict, name="bm25"))
    vector_run = min_max_norm_inverted(Run(vector_dict, name="vector"))
    print("Before fusion:")
    compare_search_results(qrels, [bm25_run, vector_run])
    print("\nAfter fusion:")
    search_results_fusion(qrels, [bm25_run, vector_run], fusion_methods_with_optimize)


# #    Model               MRR@10              MAP@10              NDCG@10
# ---  ------------------  ------------------  ------------------  -------------------
# a    bm25                0.173               0.170               0.216
# b    vector              0.220ᵃ              0.215ᵃ              0.263ᵃ
# c    rrf                 0.252ᵃᵇᵉᶠᵍᵏˡᵐⁿᵒ     0.247ᵃᵇᵉᶠᵍᵏˡᵐⁿᵒ     0.306ᵃᵇᵉᶠᵍᵏˡᵐⁿᵒ
# d    comb_gmnz           0.259ᵃᵇᶜᵉᶠᵍᶦʲᵏˡᵐⁿᵒ  0.254ᵃᵇᶜᵉᶠᵍᶦʲᵏˡᵐⁿᵒ  0.314ᵃᵇᶜᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒ
# e    probfuse            0.233ᵃᵇᵒ            0.228ᵃᵇᵒ            0.286ᵃᵇᵒ
# f    slidefuse           0.246ᵃᵇᵉᵏᵐᵒ         0.242ᵃᵇᵉᵏᵐᵒ         0.300ᵃᵇᵉᵏˡᵐⁿᵒ
# g    bayesfuse           0.242ᵃᵇᵉᵒ           0.238ᵃᵇᵉᵒ           0.297ᵃᵇᵉᵒ
# h    wmnz                0.258ᵃᵇᶜᵉᶠᵍᶦᵏˡᵐⁿᵒ   0.253ᵃᵇᶜᵉᶠᵍᶦᵏˡᵐⁿᵒ   0.311ᵃᵇᶜᵉᶠᵍᶦᵏˡᵐⁿᵒ
# i    rbc                 0.249ᵃᵇᵉᵍᵏˡᵐⁿᵒ      0.245ᵃᵇᵉᵍᵏˡᵐⁿᵒ      0.305ᵃᵇᵉᶠᵍᵏˡᵐⁿᵒ
# j    logn_isr            0.254ᵃᵇᵉᶠᵍᵏˡᵐⁿᵒ     0.249ᵃᵇᵉᶠᵍᵏˡᵐⁿᵒ     0.309ᵃᵇᵉᶠᵍᵏˡᵐⁿᵒ
# k    posfuse             0.239ᵃᵇᵉᵐᵒ          0.234ᵃᵇᵉᵐᵒ          0.292ᵃᵇᵉᵒ
# l    segfuse             0.242ᵃᵇᵉᵐᵒ          0.238ᵃᵇᵉᵐᵒ          0.294ᵃᵇᵉᵒ
# m    mapfuse             0.237ᵃᵇᵉᵒ           0.232ᵃᵇᵉᵒ           0.291ᵃᵇᵉᵒ
# n    weighted_bordafuse  0.244ᵃᵇᵉᵒ           0.240ᵃᵇᵉᵒ           0.292ᵃᵇᵒ
# o    weighted_condorcet  0.220ᵃ              0.215ᵃ              0.263ᵃ
# p    mixed               0.259ᵃᵇᶜᵉᶠᵍᶦᵏˡᵐⁿᵒ   0.254ᵃᵇᶜᵉᶠᵍᶦʲᵏˡᵐⁿᵒ  0.313ᵃᵇᶜᵉᶠᵍᶦᵏˡᵐⁿᵒ
# q    weighted_sum        0.258ᵃᵇᶜᵉᶠᵍᶦᵏˡᵐⁿᵒ   0.253ᵃᵇᶜᵉᶠᵍᶦᵏˡᵐⁿᵒ   0.312ᵃᵇᶜᵉᶠᵍᶦᵏˡᵐⁿᵒ
