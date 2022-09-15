import numpy as np

# k即为NDCG@k

def hr(rank, k):
    res = 0.0
    for r in rank:
        if r < k:
            res += 1
    return res / len(rank)


def mrr(rank, k):
    mrr = 0.0
    for r in rank:
        if r < k:
            mrr += 1 / (r + 1)
    return mrr / len(rank)


def ndcg(rank, k):
    res = 0.0
    for r in rank:
        if r < k:
            res += 1 / np.log2(r + 2)
    return res / len(rank)
