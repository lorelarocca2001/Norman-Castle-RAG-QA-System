# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:10:23 2026

@author: loren
"""

from src.loader import load_reader

def extract_answers(question, retrieved_chunks):
    qa_reader = load_reader()

    candidate_answers = []

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        result = qa_reader(
            question=question,
            context=chunk["context"],
            handle_impossible_answer=True
        )

        answer = result["answer"].strip()
        no_answer = answer == ""

        candidate_answers.append({
            "answer": answer,
            "score": float(result["score"]),
            "start": int(result["start"]),
            "end": int(result["end"]),
            "reranked_rank": rank,
            "chunk_id": chunk["chunk_id"],
            "section_title": chunk["section_title"],
            "retrieval_score": chunk["score"],
            "reranker_score": chunk["reranker_score"],
            "combined_score": None,
            "bert_rank_score": None,
            "context": chunk["context"],
            "no_answer": no_answer
        })

    return candidate_answers


def extract_answers_reranked(question, retrieved_chunks):
    qa_reader = load_reader()

    questions = [question for _ in retrieved_chunks]
    contexts = [chunk["context"] for chunk in retrieved_chunks]

    raw_results = qa_reader(
        question=questions,
        context=contexts,
        handle_impossible_answer=True,
        batch_size=len(retrieved_chunks),
        top_k=2
    )

    candidate_answers = []

    for rank, (chunk, result_group) in enumerate(zip(retrieved_chunks, raw_results), start=1):
        if isinstance(result_group, dict):
            result_group = [result_group]

        best_result = result_group[0]
        answer = best_result["answer"].strip()

        reader_score = float(best_result["score"])
        retrieval_score = float(chunk["score"])
        reranker_score = float(chunk["reranker_score"])
        combined_score = retrieval_score * reranker_score
        bert_rank_score = reader_score * reranker_score

        second_valid_result = None

        for result in result_group[1:]:
            if result["answer"].strip() != "":
                second_valid_result = result
                break

        candidate_answers.append({
            "answer": answer,
            "score": reader_score,
            "start": int(best_result["start"]),
            "end": int(best_result["end"]),
            "reranked_rank": rank,
            "chunk_id": chunk["chunk_id"],
            "section_title": chunk["section_title"],
            "retrieval_score": retrieval_score,
            "reranker_score": reranker_score,
            "combined_score": combined_score,
            "bert_rank_score": bert_rank_score,
            "context": chunk["context"],
            "no_answer": answer == "",
            "second_valid_answer": second_valid_result
        })

    return candidate_answers