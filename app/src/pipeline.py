# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:10:42 2026

@author: loren
"""

from src.config import (
    FAISS_TOP_K,
    RERANK_TOP_K,
    EXTRACTIVE_READER_TOP_K,
    GENERATIVE_TOP_K,
    MIN_READER_SCORE,
    MIN_BERT_RANK_SCORE,
    MIN_FORCED_PRODUCT_SCORE,
    MAX_EMPTY_ANSWER_SCORE,
    NO_ANSWER_TEXT
)

from src.retriever import retrieve_faiss
from src.reranker import retrieve_faiss_reranked
from src.reader import extract_answers, extract_answers_reranked
from src.generative_answerer import generate_answer


def retrieve_chunks(
    question,
    faiss_top_k,
    final_top_k,
    use_reranker=True
):
    if use_reranker:
        return retrieve_faiss_reranked(
            question=question,
            faiss_top_k=faiss_top_k,
            rerank_top_k=final_top_k
        )

    retrieved_chunks = retrieve_faiss(
        question=question,
        top_k=final_top_k
    )

    for chunk in retrieved_chunks:
        chunk["reranker_score"] = None

    return retrieved_chunks


def answer_question_extractive_simple(
    question,
    faiss_top_k=FAISS_TOP_K,
    reader_top_k=EXTRACTIVE_READER_TOP_K,
    min_reader_score=MIN_READER_SCORE
):
    retrieved_chunks = retrieve_chunks(
        question=question,
        faiss_top_k=faiss_top_k,
        final_top_k=reader_top_k,
        use_reranker=False
    )

    if not retrieved_chunks:
        return None, []

    candidate_answers = extract_answers(
        question=question,
        retrieved_chunks=retrieved_chunks
    )

    if not candidate_answers:
        return None, []

    valid_answers = [
        candidate
        for candidate in candidate_answers
        if not candidate.get("no_answer", False)
    ]

    if len(valid_answers) == 0:
        best_empty = max(candidate_answers, key=lambda x: x["score"])

        best_answer = {
            "answer": NO_ANSWER_TEXT,
            "raw_answer": "",
            "score": best_empty["score"],
            "start": None,
            "end": None,
            "reranked_rank": best_empty["reranked_rank"],
            "chunk_id": best_empty["chunk_id"],
            "section_title": best_empty["section_title"],
            "retrieval_score": best_empty["retrieval_score"],
            "reranker_score": best_empty["reranker_score"],
            "combined_score": None,
            "bert_rank_score": None,
            "context": best_empty["context"],
            "no_answer": True,
            "forced_by_retrieval_reranker": False,
            "selected_by_bert_rank_score": False,
            "rejected_by_bert_rank_score": False
        }

        return best_answer, candidate_answers

    best_answer = max(valid_answers, key=lambda x: x["score"])

    if best_answer["score"] < min_reader_score:
        best_answer = best_answer.copy()
        best_answer["answer"] = NO_ANSWER_TEXT
        best_answer["raw_answer"] = best_answer.get("raw_answer", "")
        best_answer["start"] = None
        best_answer["end"] = None
        best_answer["no_answer"] = True
        best_answer["forced_by_retrieval_reranker"] = False
        best_answer["selected_by_bert_rank_score"] = False
        best_answer["rejected_by_bert_rank_score"] = False

        return best_answer, candidate_answers

    best_answer["no_answer"] = False
    best_answer["forced_by_retrieval_reranker"] = False
    best_answer["selected_by_bert_rank_score"] = False
    best_answer["rejected_by_bert_rank_score"] = False

    return best_answer, candidate_answers


def answer_question_extractive_reranked(
    question,
    faiss_top_k=FAISS_TOP_K,
    reader_top_k=EXTRACTIVE_READER_TOP_K,
    min_bert_rank_score=MIN_BERT_RANK_SCORE,
    min_forced_product_score=MIN_FORCED_PRODUCT_SCORE,
    max_empty_answer_score=MAX_EMPTY_ANSWER_SCORE
):
    retrieved_chunks = retrieve_chunks(
        question=question,
        faiss_top_k=faiss_top_k,
        final_top_k=reader_top_k,
        use_reranker=True
    )

    if not retrieved_chunks:
        return None, []

    candidate_answers = extract_answers_reranked(
        question=question,
        retrieved_chunks=retrieved_chunks
    )

    if not candidate_answers:
        return None, []

    valid_answers = [
        candidate
        for candidate in candidate_answers
        if candidate["answer"] != ""
    ]

    if len(valid_answers) == 0:
        best_forced_candidate = max(
            candidate_answers,
            key=lambda x: x["combined_score"]
        )

        second_valid_result = best_forced_candidate["second_valid_answer"]

        can_force_answer = (
            second_valid_result is not None
            and best_forced_candidate["combined_score"] >= min_forced_product_score
            and best_forced_candidate["score"] <= max_empty_answer_score
        )

        if can_force_answer:
            forced_reader_score = float(second_valid_result["score"])
            forced_bert_rank_score = forced_reader_score * best_forced_candidate["reranker_score"]

            return {
                "answer": second_valid_result["answer"].strip(),
                "score": forced_reader_score,
                "start": int(second_valid_result["start"]),
                "end": int(second_valid_result["end"]),
                "reranked_rank": best_forced_candidate["reranked_rank"],
                "chunk_id": best_forced_candidate["chunk_id"],
                "section_title": best_forced_candidate["section_title"],
                "retrieval_score": best_forced_candidate["retrieval_score"],
                "reranker_score": best_forced_candidate["reranker_score"],
                "combined_score": best_forced_candidate["combined_score"],
                "bert_rank_score": forced_bert_rank_score,
                "empty_answer_score": best_forced_candidate["score"],
                "context": best_forced_candidate["context"],
                "no_answer": False,
                "forced_by_retrieval_reranker": True,
                "selected_by_bert_rank_score": False,
                "rejected_by_bert_rank_score": False
            }, candidate_answers

        return {
            "answer": NO_ANSWER_TEXT,
            "score": best_forced_candidate["score"],
            "start": None,
            "end": None,
            "reranked_rank": best_forced_candidate["reranked_rank"],
            "chunk_id": best_forced_candidate["chunk_id"],
            "section_title": best_forced_candidate["section_title"],
            "retrieval_score": best_forced_candidate["retrieval_score"],
            "reranker_score": best_forced_candidate["reranker_score"],
            "combined_score": best_forced_candidate["combined_score"],
            "bert_rank_score": best_forced_candidate["bert_rank_score"],
            "empty_answer_score": best_forced_candidate["score"],
            "context": best_forced_candidate["context"],
            "no_answer": True,
            "forced_by_retrieval_reranker": False,
            "selected_by_bert_rank_score": False,
            "rejected_by_bert_rank_score": False
        }, candidate_answers

    best_answer = max(valid_answers, key=lambda x: x["bert_rank_score"])
    selected_by_bert_rank_score = len(valid_answers) >= 2

    if best_answer["bert_rank_score"] < min_bert_rank_score:
        return {
            "answer": NO_ANSWER_TEXT,
            "score": best_answer["score"],
            "start": None,
            "end": None,
            "reranked_rank": best_answer["reranked_rank"],
            "chunk_id": best_answer["chunk_id"],
            "section_title": best_answer["section_title"],
            "retrieval_score": best_answer["retrieval_score"],
            "reranker_score": best_answer["reranker_score"],
            "combined_score": best_answer["combined_score"],
            "bert_rank_score": best_answer["bert_rank_score"],
            "context": best_answer["context"],
            "no_answer": True,
            "forced_by_retrieval_reranker": False,
            "selected_by_bert_rank_score": False,
            "rejected_by_bert_rank_score": True
        }, candidate_answers

    best_answer["no_answer"] = False
    best_answer["forced_by_retrieval_reranker"] = False
    best_answer["selected_by_bert_rank_score"] = selected_by_bert_rank_score
    best_answer["rejected_by_bert_rank_score"] = False

    return best_answer, candidate_answers


def answer_question_extractive(
    question,
    faiss_top_k=FAISS_TOP_K,
    rerank_top_k=RERANK_TOP_K,
    reader_top_k=EXTRACTIVE_READER_TOP_K,
    use_reranker=True,
    min_reader_score=MIN_READER_SCORE,
    min_bert_rank_score=MIN_BERT_RANK_SCORE,
    min_forced_product_score=MIN_FORCED_PRODUCT_SCORE,
    max_empty_answer_score=MAX_EMPTY_ANSWER_SCORE
):
    if use_reranker:
        final_reader_top_k = min(reader_top_k, rerank_top_k)

        return answer_question_extractive_reranked(
            question=question,
            faiss_top_k=faiss_top_k,
            reader_top_k=final_reader_top_k,
            min_bert_rank_score=min_bert_rank_score,
            min_forced_product_score=min_forced_product_score,
            max_empty_answer_score=max_empty_answer_score
        )

    return answer_question_extractive_simple(
        question=question,
        faiss_top_k=faiss_top_k,
        reader_top_k=reader_top_k,
        min_reader_score=min_reader_score
    )


def answer_question_generative(
    question,
    faiss_top_k=FAISS_TOP_K,
    generative_top_k=GENERATIVE_TOP_K,
    use_reranker=True
):
    retrieved_chunks = retrieve_chunks(
        question=question,
        faiss_top_k=faiss_top_k,
        final_top_k=generative_top_k,
        use_reranker=use_reranker
    )

    if not retrieved_chunks:
        return None, []

    result = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks
    )

    best_source = result["sources"][0]

    if best_source["reranker_score"] is not None:
        score = best_source["reranker_score"]
    else:
        score = best_source["retrieval_score"]

    best_answer = {
        "answer": result["answer"],
        "score": score,
        "reranked_rank": 1,
        "chunk_id": best_source["chunk_id"],
        "section_title": best_source["section_title"],
        "retrieval_score": best_source["retrieval_score"],
        "reranker_score": best_source["reranker_score"],
        "combined_score": None,
        "bert_rank_score": None,
        "context": best_source["context"],
        "sources": result["sources"]
    }

    return best_answer, result["sources"]


def answer_question(
    question,
    mode="Estrattiva",
    faiss_top_k=FAISS_TOP_K,
    rerank_top_k=RERANK_TOP_K,
    extractive_reader_top_k=EXTRACTIVE_READER_TOP_K,
    generative_top_k=GENERATIVE_TOP_K,
    use_reranker=True,
    min_reader_score=MIN_READER_SCORE,
    min_bert_rank_score=MIN_BERT_RANK_SCORE,
    min_forced_product_score=MIN_FORCED_PRODUCT_SCORE,
    max_empty_answer_score=MAX_EMPTY_ANSWER_SCORE
):
    if mode == "Generativa":
        return answer_question_generative(
            question=question,
            faiss_top_k=faiss_top_k,
            generative_top_k=generative_top_k,
            use_reranker=use_reranker
        )

    return answer_question_extractive(
        question=question,
        faiss_top_k=faiss_top_k,
        rerank_top_k=rerank_top_k,
        reader_top_k=extractive_reader_top_k,
        use_reranker=use_reranker,
        min_reader_score=min_reader_score,
        min_bert_rank_score=min_bert_rank_score,
        min_forced_product_score=min_forced_product_score,
        max_empty_answer_score=max_empty_answer_score
    )