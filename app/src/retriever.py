# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:09:35 2026

@author: loren
"""


from src.loader import load_embedding_model, load_faiss_index, load_metadata


def retrieve_faiss(question, top_k=5):
    model = load_embedding_model()
    index = load_faiss_index()
    metadata = load_metadata()

    question_embedding = model.encode(
        ["query: " + question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(question_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        item = metadata[idx]

        results.append({
            "chunk_id": item["chunk_id"],
            "section_title": item["section_title"],
            "score": float(score),
            "context": item["text"]
        })

    return results