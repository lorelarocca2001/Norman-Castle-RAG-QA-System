# RAG-based Question Answering System for the Norman Castle of Aci Castello

This project implements a Retrieval-Augmented Generation system for answering questions about a historical text related to the Norman Castle of Aci Castello.

The system uses semantic embeddings, FAISS vector search, optional reranking, and both extractive and generative answer generation. The application is developed in Python with Streamlit and deployed on Hugging Face Spaces.

## Main Features

- Text chunking and semantic embedding generation
- FAISS-based vector retrieval
- Optional reranking of retrieved chunks
- Extractive Question Answering with a BERT-based reader
- Generative answer generation with an LLM
- Evaluation on a SQuAD-like dataset
- Streamlit web interface
- Deployment on Hugging Face Spaces

## Webapp

The deployed webapp is available at:

[[Hugging Face Space link here](https://huggingface.co/spaces/lorelarocca2001/guida-castello-aci)]

## Repository Structure

- `app/`: Streamlit webapp source code
- `notebooks/`: development and experimentation notebook
- `data/`: chunks and QA evaluation dataset
- `indexes/`: FAISS index and metadata
- `assets/`: screenshots and architecture images
- `docs/`: additional project documentation

## How to Run

```bash
pip install -r requirements.txt
streamlit run app/main.py