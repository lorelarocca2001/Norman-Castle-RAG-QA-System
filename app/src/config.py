from pathlib import Path
import torch
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "indexes"
OUTPUT_DIR = BASE_DIR / "outputs"

FAISS_INDEX_PATH = DATA_DIR / "faiss_chunks_e5.index"
METADATA_PATH = DATA_DIR / "faiss_chunks_metadata_e5.json"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
READER_MODEL_NAME = "deepset/xlm-roberta-base-squad2"

FAISS_TOP_K = 10
RERANK_TOP_K = 3
EXTRACTIVE_READER_TOP_K = 3
GENERATIVE_TOP_K = 3

MIN_READER_SCORE = 0.30
MIN_BERT_RANK_SCORE = 0.25
MIN_FORCED_PRODUCT_SCORE = 0.68
MAX_EMPTY_ANSWER_SCORE = 0.55

NO_ANSWER_TEXT = "La risposta non è presente nei documenti disponibili."

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRANSFORMERS_DEVICE = 0 if torch.cuda.is_available() else -1

WHISPER_MODEL_SIZE = "small"
WHISPER_LANGUAGE = "it"

OUTPUT_DIR.mkdir(exist_ok=True)