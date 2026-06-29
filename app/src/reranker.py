# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:10:06 2026

@author: loren
"""

import numpy as np

from src.loader import load_reranker
from src.retriever import retrieve_faiss


def retrieve_faiss_reranked(question, faiss_top_k=10, rerank_top_k=3):
    initial_results = retrieve_faiss(
        question=question,
        top_k=faiss_top_k
    )

    if not initial_results:
        return []

    reranker = load_reranker()

    pairs = [
        [question, result["context"]]
        for result in initial_results
    ]

    reranker_logits = reranker.predict(pairs)
    reranker_scores = 1 / (1 + np.exp(-reranker_logits))

    reranked_results = []

    for result, reranker_score, reranker_logit in zip(initial_results, reranker_scores, reranker_logits):
        item = result.copy()
        item["reranker_score"] = float(reranker_score)
        item["reranker_logit"] = float(reranker_logit)
        reranked_results.append(item)

    reranked_results = sorted(
        reranked_results,
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return reranked_results[:rerank_top_k]