import json
import faiss
import streamlit as st

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from faster_whisper import WhisperModel

from src.config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    READER_MODEL_NAME,
    TRANSFORMERS_DEVICE,
    DEVICE,
    WHISPER_MODEL_SIZE
)


@st.cache_resource(show_spinner="Caricamento dell'indice FAISS...")
def load_faiss_index():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"Indice FAISS non trovato: {FAISS_INDEX_PATH}")

    return faiss.read_index(str(FAISS_INDEX_PATH))


@st.cache_resource(show_spinner="Caricamento dei metadati...")
def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"File metadati non trovato: {METADATA_PATH}")

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Caricamento del modello di embedding...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Caricamento del reranker...")
def load_reranker():
    return CrossEncoder(
        RERANKER_MODEL_NAME,
        device=DEVICE
    )


@st.cache_resource(show_spinner="Caricamento del reader estrattivo...")
def load_reader():
    tokenizer = AutoTokenizer.from_pretrained(
        READER_MODEL_NAME,
        use_fast=False
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        READER_MODEL_NAME
    )

    return pipeline(
        task="question-answering",
        model=model,
        tokenizer=tokenizer,
        device=TRANSFORMERS_DEVICE,
        handle_impossible_answer=True
    )


@st.cache_resource(show_spinner="Caricamento del modello Whisper...")
def load_whisper_model(whisper_model_size=WHISPER_MODEL_SIZE):
    compute_type = "float16" if DEVICE == "cuda" else "int8"

    return WhisperModel(
        whisper_model_size,
        device=DEVICE,
        compute_type=compute_type
    )