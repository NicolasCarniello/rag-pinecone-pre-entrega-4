"""Configuración central. Todo se puede sobreescribir con variables de entorno (.env).

Las credenciales NUNCA se escriben en el código: se leen de PINECONE_API_KEY y
GOOGLE_API_KEY (u OPENAI_API_KEY si se usa OpenAI).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
GOLDEN_SET = Path(os.getenv("GOLDEN_SET", BASE_DIR / "golden_set.json"))

# --- Pinecone ----------------------------------------------------------------
INDEX_NAME = os.getenv("INDEX_NAME", "andespay-docs")
# Namespace por entorno: los datos de prueba nunca se mezclan con los de producción.
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "dev")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")  # región del plan gratuito
METRIC = "cosine"  # métrica recomendada para embeddings de texto

# --- Embeddings ----------------------------------------------------------------
# La dimensión se define en UN solo lugar y se usa para crear el índice y para
# validar cada vector antes de subirlo (evita el "mismatch de dimensiones").
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
_DEFAULT_EMB = {"gemini": "gemini-embedding-001", "openai": "text-embedding-3-small"}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", _DEFAULT_EMB.get(EMBEDDING_PROVIDER, ""))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# --- LLM (opcional: solo para generar una respuesta con los documentos recuperados)
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-flash-latest")

# --- Chunking (en tokens) -------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))  # punto medio del rango 500–800
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# --- Ingesta y recuperación ---------------------------------------------------
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "100"))  # 100–200 recomendado
TOP_K = int(os.getenv("TOP_K", "5"))  # documentos que devuelve el sistema
# Candidatos que trae CADA recuperador antes de fusionar. Con 5 por lado y pesos
# 0.4/0.6, un resultado que solo encuentra BM25 nunca supera a uno que solo
# encuentra la búsqueda semántica (ver README), así que se piden 10 y se corta en 5.
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "10"))
# Pesos del EnsembleRetriever: [BM25 (léxico), Pinecone (semántico)]
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))

_KEY_FOR = {"gemini": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}


def require_env(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise RuntimeError(
            f"Falta la variable de entorno {', '.join(missing)}. "
            "Copiá .env.example a .env y completala (el .env no se sube a Git)."
        )


def get_embeddings():
    """Modelo de embeddings, siempre con la dimensión EMBEDDING_DIM."""
    require_env(_KEY_FOR[EMBEDDING_PROVIDER])
    if EMBEDDING_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        # gemini-embedding-001 admite 768, 1536 o 3072 dimensiones (Matryoshka).
        # Usa task_type RETRIEVAL_DOCUMENT al indexar y RETRIEVAL_QUERY al consultar.
        return GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL, output_dimensionality=EMBEDDING_DIM
        )
    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIM)
    raise ValueError(f"EMBEDDING_PROVIDER desconocido: {EMBEDDING_PROVIDER}")


def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    require_env("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=None, max_retries=3)


def get_pinecone():
    from pinecone import Pinecone

    require_env("PINECONE_API_KEY")
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])
