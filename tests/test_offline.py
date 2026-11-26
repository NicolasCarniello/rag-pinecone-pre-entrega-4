"""Tests offline (sin cuentas ni red): Pinecone falso + embeddings deterministas.

    pytest -q
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fake_pinecone import FakeIndex, FakePinecone
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.retrievers import BaseRetriever
from langchain_pinecone import PineconeVectorStore
from pydantic import ValidationError

import config
import evaluate
from ingest import chunk_sources, count_tokens, ingest, load_sources, upsert_chunks
from rag_system import EnsembleRetriever, RAGSystem, load_corpus, tokenize_es
from schemas import build_metadata
from setup_index import IndexConfigError, ensure_index

ROOT = Path(__file__).resolve().parents[1]
NS = "test"


@pytest.fixture
def emb():
    return DeterministicFakeEmbedding(size=config.EMBEDDING_DIM)


@pytest.fixture
def index():
    return FakeIndex(config.EMBEDDING_DIM)


@pytest.fixture
def loaded(index, emb):
    ingest(index, emb, namespace=NS, wait_seconds=0)
    return index


@pytest.fixture
def rag(loaded, emb):
    vs = PineconeVectorStore(index=loaded, embedding=emb, text_key="text", namespace=NS)
    return RAGSystem(vs, load_corpus(loaded, NS))


# ---------------------------------------------------------------------------
# 1. Infraestructura
# ---------------------------------------------------------------------------
def test_crea_indice_serverless_si_no_existe():
    pc = FakePinecone()
    idx = ensure_index(pc, "docs", poll=0)
    created = pc.created[0]
    assert created["dimension"] == config.EMBEDDING_DIM and created["metric"] == "cosine"
    assert created["spec"].cloud == "aws" and created["spec"].region == "us-east-1"
    assert isinstance(idx, FakeIndex)


def test_no_recrea_indice_existente():
    pc = FakePinecone()
    ensure_index(pc, "docs", poll=0)
    ensure_index(pc, "docs", poll=0)
    assert len(pc.created) == 1


def test_detecta_mismatch_de_dimension():
    pc = FakePinecone()
    ensure_index(pc, "docs", dimension=768, poll=0)
    with pytest.raises(IndexConfigError, match="dimensión 768"):
        ensure_index(pc, "docs", dimension=1536, poll=0)


def test_credenciales_fuera_del_repo():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for var in ("PINECONE_API_KEY", "GOOGLE_API_KEY", "INDEX_NAME"):
        assert re.search(rf"^{var}=\s*$", example, re.M) or re.search(rf"^{var}=[\w-]+$", example, re.M)
    pattern = re.compile(r"(pcsk_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_\-]{30,}|sk-[A-Za-z0-9]{20,})")
    try:
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                               check=True).stdout.split()
    except Exception:
        files = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()]
    for f in files:
        p = ROOT / f
        if p.name == ".env" or ".venv" in p.parts or p.suffix in {".pdf", ".pyc"} or not p.exists():
            continue
        assert not pattern.search(p.read_text(encoding="utf-8", errors="ignore")), f"posible clave en {f}"


# ---------------------------------------------------------------------------
# 2. Ingesta y metadatos
# ---------------------------------------------------------------------------
def test_loaders_de_los_tres_formatos():
    sources = {s.doc_id: s for s in load_sources()}
    assert {s.source_type for s in sources.values()} >= {"markdown", "json", "pdf"}
    assert sources["pagos"].category == "referencia-api"  # frontmatter
    assert sources["errores"].category == "errores"  # campo del JSON
    assert "AP-4290" in sources["errores"].pages[0][1]
    assert [p for p, _ in sources["changelog"].pages] == [1, 2, 3]  # PDF página por página


def test_chunks_con_metadatos_completos():
    chunks = chunk_sources(load_sources())
    required = {"text", "chunk_id", "doc_id", "source", "source_type", "title", "category",
                "tags", "page", "section", "chunk_index", "n_tokens", "char_count", "ingested_at"}
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)
    for c in chunks:
        m = c.metadata
        assert required <= m.keys()
        assert m["text"] == c.page_content and m["n_tokens"] <= config.CHUNK_SIZE
        assert all(isinstance(v, (str, int, float, bool, list)) for v in m.values())  # tipos válidos en Pinecone
    pdf = [c for c in chunks if c.metadata["source_type"] == "pdf"]
    assert {c.metadata["page"] for c in pdf} == {1, 2, 3}
    assert all(f"Página {c.metadata['page']}" in c.page_content for c in pdf)


def test_esquema_de_metadatos_valida_y_normaliza():
    m = build_metadata(text="hola", doc_id="Pagos", source="pagos.md", source_type="markdown",
                       title="t", category=" Referencia-API ", tags=["B", "a", "a"], page=0,
                       section="", chunk_index=3, n_tokens=1)
    assert m["chunk_id"] == "pagos::p0::c03" and m["category"] == "referencia-api" and m["tags"] == ["a", "b"]
    with pytest.raises(ValidationError):
        build_metadata(text="x", doc_id="d", source="d.md", source_type="docx", title="t",
                       category="c", tags=[], page=0, section="", chunk_index=0, n_tokens=1)


def test_upsert_por_lotes(index, emb):
    chunks = chunk_sources(load_sources())
    calls = upsert_chunks(index, emb, chunks, namespace=NS, batch_size=5)
    assert calls == -(-len(chunks) // 5) == index.upsert_calls
    assert index.describe_index_stats().namespaces[NS].vector_count == len(chunks)


def test_rechaza_embeddings_de_otra_dimension(index):
    with pytest.raises(ValueError, match="dimensión"):
        upsert_chunks(index, DeterministicFakeEmbedding(size=768), chunk_sources(load_sources()), NS)


def test_reingesta_idempotente_y_borra_obsoletos(tmp_path, index, emb):
    data = tmp_path / "data"
    shutil.copytree(config.DATA_DIR, data)
    first = ingest(index, emb, data_dir=data, namespace=NS, wait_seconds=0)
    again = ingest(index, emb, data_dir=data, namespace=NS, wait_seconds=0)
    assert again["vectores_en_namespace"] == first["chunks"]  # no duplica
    (data / "webhooks.md").unlink()
    after = ingest(index, emb, data_dir=data, namespace=NS, wait_seconds=0)
    assert after["obsoletos_eliminados"] > 0
    assert not any(i.startswith("webhooks::") for page in index.list(namespace=NS) for i in page)


def test_namespaces_aislados(index, emb):
    ingest(index, emb, namespace="dev", wait_seconds=0)
    assert "prod" not in index.describe_index_stats().namespaces
    assert index.query([0.1] * config.EMBEDDING_DIM, namespace="prod")["matches"] == []


# ---------------------------------------------------------------------------
# 3. Recuperador híbrido
# ---------------------------------------------------------------------------
def test_tokenizador_bm25():
    toks = tokenize_es("¿Qué significa AP-4290 en client.refunds.list y la Paginación?")
    assert {"ap-4290", "ap", "4290", "client.refunds.list", "refunds", "paginacion"} <= set(toks)
    assert "que" not in toks and "la" not in toks


def test_corpus_bm25_sale_del_indice(rag, loaded):
    assert len(rag.documents) == loaded.describe_index_stats().namespaces[NS].vector_count
    assert all(d.page_content.startswith("Documento:") for d in rag.documents)


def test_rag_devuelve_top5_hibrido(rag):
    docs = rag.retrieve("¿Cómo pagino los reembolsos con cursor?")
    assert len(docs) == 5
    assert [d.metadata["rank"] for d in docs] == [1, 2, 3, 4, 5]
    assert len({d.metadata["chunk_id"] for d in docs}) == 5  # sin duplicados entre BM25 y Pinecone


def test_ensemble_usa_bm25_y_pinecone_con_pesos(rag):
    ens = rag.ensemble()
    assert [type(r).__name__ for r in ens.retrievers] == ["BM25Retriever", "VectorStoreRetriever"]
    assert ens.weights == [0.4, 0.6] and ens.id_key == "chunk_id"
    assert ens.retrievers[0].k == ens.retrievers[1].search_kwargs["k"] == config.CANDIDATE_K


class _Fija(BaseRetriever):
    """Recuperador con un ranking fijo, para probar la fusión de forma determinística."""
    ids: list[str]

    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=i, metadata={"chunk_id": i}) for i in self.ids]


def _top5(bm25_ids, sem_ids):
    ens = EnsembleRetriever(retrievers=[_Fija(ids=bm25_ids), _Fija(ids=sem_ids)],
                            weights=[0.4, 0.6], id_key="chunk_id")
    return [d.metadata["chunk_id"] for d in ens.invoke("q")][:5]


def test_rrf_con_5_candidatos_bm25_solo_no_entra_al_top5():
    """X es el 1.º de BM25 y la búsqueda semántica lo tiene 8.º.
    Con 5 candidatos por lado, la semántica no lo trae y X queda afuera:
    0.4/61 ≈ 0.0066 < 0.6/65 ≈ 0.0092 (el 5.º semántico)."""
    sem = [f"s{i}" for i in range(1, 8)] + ["X", "s9", "s10"]
    assert "X" not in _top5(["X", "a", "b", "c", "d"], sem[:5])
    # Con 10 candidatos la semántica también lo ve: 0.4/61 + 0.6/68 ≈ 0.0154 → pasa al 1.º lugar.
    assert _top5(["X", "a", "b", "c", "d", "e", "f", "g", "h", "i"], sem)[0] == "X"


def test_bm25_encuentra_codigos_exactos(rag):
    top = rag.retrieve("¿Qué significa el código AP-4290?", mode="bm25")[0]
    assert top.metadata["doc_id"] == "errores"
    top = rag.retrieve("PaymentIntent.confirm deprecado", mode="bm25")[0]
    assert top.metadata["doc_id"] == "changelog"


@pytest.mark.parametrize("mode", ["hybrid", "semantic", "bm25"])
def test_filtro_duro_por_categoria(rag, mode):
    docs = rag.retrieve("errores y excepciones", mode=mode, category="errores")
    assert docs and all(d.metadata["category"] == "errores" for d in docs)


def test_consulta_vacia(rag):
    with pytest.raises(ValueError):
        rag.retrieve("  ")


# ---------------------------------------------------------------------------
# 4. Evaluación
# ---------------------------------------------------------------------------
def test_metricas_ejemplo_de_la_clase():
    # 5 recuperados, 2 relevantes entre ellos, 4 relevantes en total → P@5 = 0.4, R@5 = 0.5
    retrieved = ["a", "x", "b", "y", "z"]
    relevant = {"a", "b", "c", "d"}
    assert evaluate.precision_at_k(retrieved, relevant) == pytest.approx(0.4)
    assert evaluate.recall_at_k(retrieved, relevant) == pytest.approx(0.5)
    assert evaluate.reciprocal_rank(retrieved, relevant) == 1.0


def test_metricas_por_pregunta():
    m = evaluate.evaluate_question(["pagos", "errores", "pagos", "webhooks", "changelog"],
                                   ["pagos::c0", "e::c0", "pagos::c1", "w::c0", "c::c0"],
                                   "pagos", {"pagos::c0", "pagos::c1"})
    assert m == {"recall@k": 1.0, "precision@k": 0.4, "precision_max": 0.4,
                 "recall_chunks@k": 1.0, "mrr": 1.0}
    miss = evaluate.evaluate_question(["errores"] * 5, ["e"] * 5, "pagos", {"pagos::c0"})
    assert miss["recall@k"] == 0 and miss["precision@k"] == 0 and miss["mrr"] == 0


def test_golden_set_valido():
    golden = evaluate.load_golden_set()
    assert len(golden) >= 5
    doc_ids = {s.doc_id for s in load_sources()}
    assert {g["documento_id_esperado"] for g in golden} <= doc_ids


def test_evaluacion_completa_y_readme(rag, tmp_path):
    report = evaluate.run(rag, evaluate.load_golden_set())
    assert set(report["modos"]) == {"hybrid", "hybrid_k5", "semantic", "bm25"}
    for data in report["modos"].values():
        p = data["promedios"]
        assert 0 <= p["precision@k"] <= p["precision_max"] <= 1 and 0 <= p["recall@k"] <= 1
    report.update(fecha="hoy", indice="i", namespace=NS, embeddings="fake", n_chunks=len(rag.documents))
    readme = tmp_path / "README.md"
    readme.write_text(f"antes\n{evaluate.README_START}\nviejo\n{evaluate.README_END}\ndespués")
    assert evaluate.update_readme(evaluate.markdown_summary(report), readme)
    text = readme.read_text()
    assert "viejo" not in text and "| Híbrido" in text and text.endswith("después")
    json.dumps(report)  # serializable para evidencia/metricas.json


def test_count_tokens():
    assert count_tokens("hola mundo") >= 1
