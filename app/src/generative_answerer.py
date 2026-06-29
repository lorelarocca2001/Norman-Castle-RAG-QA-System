# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 23:54:14 2026

@author: loren
"""

import os
from openai import OpenAI


def build_context(retrieved_chunks):
    blocks = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        block = (
            f"[Fonte {i}]\n"
            f"Chunk ID: {chunk['chunk_id']}\n"
            f"Sezione: {chunk['section_title']}\n"
            f"Testo:\n{chunk['context']}"
        )

        blocks.append(block)

    return "\n\n".join(blocks)


def generate_answer(question, retrieved_chunks):
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT non configurata nel file .env")

    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY non configurata nel file .env")

    if not deployment:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT non configurata nel file .env")

    client = OpenAI(
        base_url=endpoint,
        api_key=api_key
    )

    context = build_context(retrieved_chunks)

    system_prompt = """
Sei una guida turistica virtuale specializzata nella storia del Castello di Aci Castello.

Devi rispondere usando esclusivamente il contesto fornito.
Non usare conoscenza esterna.
Non inventare informazioni.

Se la risposta non è presente nel contesto recuperato, rispondi esattamente:
Non ho trovato questa informazione nei documenti recuperati.

La risposta deve essere in italiano, naturale, chiara e adatta a essere letta ad alta voce.
Non usare markdown.
Non usare asterischi.
Non usare grassetto.
Non usare elenchi puntati se non sono necessari.
"""

    user_prompt = f"""
DOMANDA:
{question}

CONTESTO RECUPERATO:
{context}

Rispondi alla domanda usando solo il contesto recuperato.
"""

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": system_prompt.strip()
            },
            {
                "role": "user",
                "content": user_prompt.strip()
            }
        ],
        max_completion_tokens=350
    )

    answer = response.choices[0].message.content.strip()

    sources = []

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        sources.append({
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "section_title": chunk["section_title"],
            "retrieval_score": chunk["score"],
            "reranker_score": chunk["reranker_score"],
            "context": chunk["context"]
        })

    return {
        "answer": answer,
        "sources": sources,
        "context": context
    }