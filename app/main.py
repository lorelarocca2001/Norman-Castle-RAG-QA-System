# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 00:46:44 2026

@author: loren
"""

import time
import hashlib
import streamlit as st
from uuid import uuid4
from audio_recorder_streamlit import audio_recorder

from src.pipeline import answer_question
from src.speech_to_text import transcribe_audio
from src.text_to_speech import synthesize_speech
from src.config import OUTPUT_DIR
from src.loader import (
    load_faiss_index,
    load_metadata,
    load_embedding_model,
    load_reranker,
    load_whisper_model
)

st.set_page_config(
    page_title="Guida virtuale - Castello di Aci",
    page_icon="🏰",
    layout="wide"
)

st.title("🏰 Guida virtuale del Castello di Aci")

st.write(
    "Benvenuto. Sono la tua guida virtuale sul Castello di Aci Castello. "
    "Puoi scrivere una domanda oppure parlare con me usando il microfono."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ciao, sono la guida virtuale del Castello di Aci. "
                "Chiedimi qualcosa sulla sua storia, sui personaggi, sulle dominazioni o sugli eventi principali."
            ),
            "audio_path": None,
            "sources": None,
            "metadata": None
        }
    ]

if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

if "processing_audio" not in st.session_state:
    st.session_state.processing_audio = False

with st.sidebar:
    st.header("Impostazioni")

    qa_mode = st.radio(
        "Modalità di risposta",
        ["Generativa", "Estrattiva"],
        horizontal=False
    )

    input_mode = st.radio(
        "Modalità di input",
        ["Testo", "Voce", "Testo e voce"],
        horizontal=False
    )

    autoplay_audio = st.checkbox(
        "Riproduci automaticamente la risposta vocale",
        value=True
    )

    show_sources_default = st.checkbox(
        "Mostra fonti aperte di default",
        value=False
    )

    with st.expander("Impostazioni avanzate", expanded=False):
        use_reranker = st.checkbox(
            "Usa reranking",
            value=True
        )

        advanced_faiss_top_k = st.number_input(
            "FAISS_TOP_K",
            min_value=1,
            max_value=30,
            value=10,
            step=1
        )

        if qa_mode == "Generativa":
            if use_reranker:
                advanced_rerank_top_k = st.number_input(
                    "RERANK_TOP_K",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1
                )
            else:
                advanced_rerank_top_k = 3

            advanced_extractive_reader_top_k = 3

            advanced_generative_top_k = st.number_input(
                "GENERATIVE_TOP_K",
                min_value=1,
                max_value=10,
                value=3,
                step=1
            )

            advanced_min_reader_score = 0.30
            advanced_min_bert_rank_score = 0.25
            advanced_min_forced_product_score = 0.68
            advanced_max_empty_answer_score = 0.55

        else:
            advanced_generative_top_k = 3

            if use_reranker:
                advanced_rerank_top_k = st.number_input(
                    "RERANK_TOP_K",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1
                )

                advanced_extractive_reader_top_k = st.number_input(
                    "EXTRACTIVE_READER_TOP_K",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1
                )

                advanced_min_bert_rank_score = st.slider(
                    "MIN_BERT_RANK_SCORE",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.25,
                    step=0.05
                )

                advanced_min_forced_product_score = st.slider(
                    "MIN_FORCED_PRODUCT_SCORE",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.68,
                    step=0.05
                )

                advanced_max_empty_answer_score = st.slider(
                    "MAX_EMPTY_ANSWER_SCORE",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.55,
                    step=0.05
                )

                advanced_min_reader_score = 0.30

                if advanced_extractive_reader_top_k > advanced_rerank_top_k:
                    st.warning(
                        "EXTRACTIVE_READER_TOP_K è maggiore di RERANK_TOP_K. "
                        "Verranno usati solo i chunk disponibili dopo il reranking."
                    )

            else:
                advanced_rerank_top_k = 3

                advanced_extractive_reader_top_k = st.number_input(
                    "EXTRACTIVE_READER_TOP_K",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1
                )

                advanced_min_reader_score = st.slider(
                    "MIN_READER_SCORE",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.30,
                    step=0.05
                )

                advanced_min_bert_rank_score = 0.25
                advanced_min_forced_product_score = 0.68
                advanced_max_empty_answer_score = 0.55

        advanced_whisper_model_size = st.selectbox(
            "WHISPER_MODEL_SIZE",
            ["tiny", "base", "small", "medium", "large-v3"],
            index=2
        )

        advanced_whisper_language = st.selectbox(
            "WHISPER_LANGUAGE",
            ["it", "en", "fr", "es", "de"],
            index=0
        )

    if st.button("Cancella conversazione"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Conversazione azzerata. "
                    "Puoi farmi una nuova domanda sul Castello di Aci."
                ),
                "audio_path": None,
                "sources": None,
                "metadata": None
            }
        ]
        st.session_state.last_audio_hash = None
        st.rerun()

with st.spinner("Caricamento della guida virtuale..."):
    load_faiss_index()
    load_metadata()
    load_embedding_model()

    if use_reranker:
        load_reranker()

    load_whisper_model(advanced_whisper_model_size)


def add_user_message(text):
    st.session_state.messages.append({
        "role": "user",
        "content": text,
        "audio_path": None,
        "sources": None,
        "metadata": None
    })


def add_assistant_message(text, audio_path=None, sources=None, metadata=None):
    st.session_state.messages.append({
        "role": "assistant",
        "content": text,
        "audio_path": audio_path,
        "sources": sources,
        "metadata": metadata
    })


def produce_answer(question):
    start_time = time.time()

    best_answer, items = answer_question(
        question=question,
        mode=qa_mode,
        faiss_top_k=advanced_faiss_top_k,
        rerank_top_k=advanced_rerank_top_k,
        extractive_reader_top_k=advanced_extractive_reader_top_k,
        generative_top_k=advanced_generative_top_k,
        use_reranker=use_reranker,
        min_reader_score=advanced_min_reader_score,
        min_bert_rank_score=advanced_min_bert_rank_score,
        min_forced_product_score=advanced_min_forced_product_score,
        max_empty_answer_score=advanced_max_empty_answer_score
    )

    elapsed_time = time.time() - start_time

    if best_answer is None:
        answer_text = "Mi dispiace, non sono riuscita a trovare una risposta nei documenti disponibili."
        audio_path = synthesize_speech(answer_text)

        add_assistant_message(
            text=answer_text,
            audio_path=audio_path,
            sources=None,
            metadata={
                "tempo": elapsed_time,
                "modalita": qa_mode
            }
        )

        return

    answer_text = best_answer["answer"]
    audio_path = synthesize_speech(answer_text)

    metadata = {
        "tempo": elapsed_time,
        "modalita": qa_mode,
        "chunk_id": best_answer["chunk_id"],
        "section_title": best_answer["section_title"],
        "retrieval_score": best_answer["retrieval_score"],
        "reranker_score": best_answer["reranker_score"],
        "score": best_answer["score"],
        "rank": best_answer["reranked_rank"],
        "context": best_answer["context"],
        "no_answer": best_answer.get("no_answer", False),
        "combined_score": best_answer.get("combined_score"),
        "bert_rank_score": best_answer.get("bert_rank_score"),
        "empty_answer_score": best_answer.get("empty_answer_score"),
        "forced_by_retrieval_reranker": best_answer.get("forced_by_retrieval_reranker", False),
        "selected_by_bert_rank_score": best_answer.get("selected_by_bert_rank_score", False),
        "rejected_by_bert_rank_score": best_answer.get("rejected_by_bert_rank_score", False)
    }

    add_assistant_message(
        text=answer_text,
        audio_path=audio_path,
        sources=items,
        metadata=metadata
    )


def render_sources(message, index):
    sources = message.get("sources")
    metadata = message.get("metadata")

    if metadata is None:
        return

    expander_title = "Fonti e dettagli della risposta"

    with st.expander(expander_title, expanded=show_sources_default):
        st.write("Modalità:", metadata.get("modalita"))
        st.write("Tempo di risposta:", f"{metadata.get('tempo', 0):.2f} secondi")

        if metadata.get("modalita") == "Estrattiva":
            if metadata.get("no_answer"):
                st.warning("Il reader non ha individuato una risposta affidabile nei chunk recuperati.")
            else:
                st.success("Il reader ha individuato una risposta estrattiva.")

        if metadata.get("section_title"):
            st.write("Fonte principale:", metadata.get("section_title"))

        if metadata.get("chunk_id"):
            st.write("Chunk principale:", metadata.get("chunk_id"))

        if metadata.get("retrieval_score") is not None:
            st.write("Score FAISS:", round(metadata.get("retrieval_score"), 4))

        if metadata.get("reranker_score") is not None:
            st.write("Score reranker:", round(metadata.get("reranker_score"), 4))

        if metadata.get("score") is not None:
            if metadata.get("modalita") == "Estrattiva":
                st.write("Score reader:", round(metadata.get("score"), 4))
            else:
                st.write("Score risposta:", round(metadata.get("score"), 4))

        if metadata.get("combined_score") is not None:
            st.write("Score FAISS × reranker:", round(metadata.get("combined_score"), 4))

        if metadata.get("bert_rank_score") is not None:
            st.write("Score BERT × reranker:", round(metadata.get("bert_rank_score"), 4))

        if metadata.get("empty_answer_score") is not None:
            st.write("Score risposta vuota:", round(metadata.get("empty_answer_score"), 4))

        if metadata.get("forced_by_retrieval_reranker"):
            st.info("Risposta forzata usando retrieval e reranker.")

        if metadata.get("selected_by_bert_rank_score"):
            st.info("Risposta selezionata usando BERT × reranker.")

        if metadata.get("rejected_by_bert_rank_score"):
            st.warning("Risposta scartata perché BERT × reranker è sotto soglia.")

        if metadata.get("rank") is not None:
            st.write("Rank:", metadata.get("rank"))

        if metadata.get("context"):
            st.markdown("Contesto principale")
            st.write(metadata.get("context"))

        if sources:
            st.markdown("Fonti recuperate")

            for source in sources:
                st.markdown("---")

                if "rank" in source:
                    st.write("Fonte:", source["rank"])

                if "reranked_rank" in source:
                    st.write("Rank reranking:", source["reranked_rank"])

                st.write("Sezione:", source.get("section_title"))
                st.write("Chunk:", source.get("chunk_id"))

                if source.get("retrieval_score") is not None:
                    st.write("Score FAISS:", round(source.get("retrieval_score"), 4))

                if source.get("reranker_score") is not None:
                    st.write("Score reranker:", round(source.get("reranker_score"), 4))

                if "answer" in source:
                    candidate_answer = source.get("answer", "")

                    if source.get("no_answer") or str(candidate_answer).strip() == "":
                        st.write("Risposta candidata:", "[NO ANSWER]")
                    else:
                        st.write("Risposta candidata:", candidate_answer)

                if source.get("score") is not None:
                    st.write("Score reader:", round(source.get("score"), 4))

                if source.get("combined_score") is not None:
                    st.write("Score FAISS × reranker:", round(source.get("combined_score"), 4))

                if source.get("bert_rank_score") is not None:
                    st.write("Score BERT × reranker:", round(source.get("bert_rank_score"), 4))

                if source.get("context"):
                    st.write(source.get("context"))


chat_container = st.container()

with chat_container:
    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            st.write(content)

            if role == "assistant" and message.get("audio_path") is not None:
                st.audio(
                    str(message["audio_path"]),
                    autoplay=autoplay_audio and i == len(st.session_state.messages) - 1
                )

            if role == "assistant":
                render_sources(message, i)

user_question = None

if input_mode in ["Testo", "Testo e voce"]:
    typed_question = st.chat_input("Scrivi una domanda alla guida virtuale...")

    if typed_question and typed_question.strip():
        user_question = typed_question.strip()

if input_mode in ["Voce", "Testo e voce"]:
    st.markdown("### Domanda vocale")

    audio_bytes = audio_recorder(
        text="Parla con la guida virtuale",
        recording_color="#e74c3c",
        neutral_color="#2ecc71",
        icon_name="microphone",
        icon_size="2x",
        pause_threshold=2.5,
        sample_rate=16000,
        auto_start=True
    )

    if audio_bytes:
        audio_hash = hashlib.md5(audio_bytes).hexdigest()

        if audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = audio_hash

            audio_path = OUTPUT_DIR / f"question_{uuid4().hex}.wav"

            with audio_path.open("wb") as f:
                f.write(audio_bytes)

            with st.spinner("Sto ascoltando la tua domanda..."):
                transcribed_question = transcribe_audio(
                    str(audio_path),
                    whisper_model_size=advanced_whisper_model_size,
                    whisper_language=advanced_whisper_language
                )

            if transcribed_question.strip():
                user_question = transcribed_question.strip()

if user_question:
    add_user_message(user_question)

    with st.spinner("La guida virtuale sta preparando la risposta..."):
        produce_answer(user_question)

    st.rerun()