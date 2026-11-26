"""Tests contra Pinecone y Gemini reales. Se saltean si faltan las claves.

    pytest -q -m integration
"""

import os

import pytest

import config

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (os.getenv("PINECONE_API_KEY") and os.getenv("GOOGLE_API_KEY")),
                       reason="requiere PINECONE_API_KEY y GOOGLE_API_KEY en .env"),
]


@pytest.fixture(scope="module")
def rag():
    from rag_system import RAGSystem

    return RAGSystem.from_pinecone()


def test_indice_con_dimension_correcta():
    pc = config.get_pinecone()
    assert pc.has_index(config.INDEX_NAME), "corré antes setup_index.py e ingest.py"
    assert pc.describe_index(config.INDEX_NAME).dimension == config.EMBEDDING_DIM


def test_namespace_poblado(rag):
    assert len(rag.documents) > 0
    assert {"markdown", "json", "pdf"} <= {d.metadata["source_type"] for d in rag.documents}


def test_hibrido_devuelve_top5_con_metadatos(rag):
    docs = rag.retrieve("¿Cómo verifico la firma de un webhook?")
    assert len(docs) == 5
    assert docs[0].metadata["doc_id"] == "webhooks"
    assert all({"source", "page", "category", "chunk_id"} <= d.metadata.keys() for d in docs)


def test_filtro_por_categoria_en_pinecone(rag):
    docs = rag.retrieve("límite de peticiones", mode="semantic", category="errores")
    assert docs and all(d.metadata["category"] == "errores" for d in docs)
