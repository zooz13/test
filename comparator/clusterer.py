from typing import Dict, List

from .embedder import cosine


def cluster_by_similarity(vectors: List[Dict[str, float]], threshold: float = 0.28) -> List[int]:
    n = len(vectors)
    if n == 0:
        return []

    adjacency = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine(vectors[i], vectors[j])
            if sim >= threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)

    cluster_ids = [-1] * n
    cid = 0
    for i in range(n):
        if cluster_ids[i] != -1:
            continue
        stack = [i]
        cluster_ids[i] = cid
        while stack:
            cur = stack.pop()
            for nxt in adjacency[cur]:
                if cluster_ids[nxt] == -1:
                    cluster_ids[nxt] = cid
                    stack.append(nxt)
        cid += 1

    return cluster_ids
