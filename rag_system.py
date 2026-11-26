"""Recuperador híbrido: BM25 (léxico) + Pinecone (semántico) con EnsembleRetriever.

Uso:
    python rag_system.py "¿Qué significa el error AP-4290?"
    python rag_system.py "¿Cómo pagino los reembolsos?" --categoria referencia-api
    python rag_system.py "¿Cómo verifico un webhook?" --responder

Por qué híbrido:
- La búsqueda semántica (embeddings) entiende sinónimos y paráfrasis, pero puede
  confundir términos exactos parecidos (AP-4091 vs AP-4100, `refunds.list` vs `payments.list`).
- BM25 premia la coincidencia exacta de términos raros: códigos de error, nombres
  de clases, funciones y parámetros.
- EnsembleRetriever combina ambos rankings con Reciprocal Rank Fusion ponderada
  (pesos 0.4 BM25 / 0.6 semántico). Cada recuperador aporta 10 candidatos y el
  sistema devuelve los top-5 de la fusión.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from typing import Literal

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

try:  # LangChain ≥ 1.0 movió los retrievers "clásicos" a langchain-classic
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:  # pragma: no cover
    from langchain.retrievers import EnsembleRetriever

import config

Mode = Literal["hybrid", "semantic", "bm25"]

# Stopwords en español: palabras sin valor para BM25.
STOPWORDS = set("""
a al algo como con cual cuales cuando cuanto cuantos de del donde el ella ellos en entre es esa
ese eso esta este esto hay la las le les lo los me mi mas no o para pero por porque que se si sin
sobre su sus te tiene tengo un una uno unos y ya yo qué cómo cuál cuándo cuánto cuántos dónde
puedo hago hacer debo uso usar
""".split())

TOKEN_RE = re.compile(r"[a-z0-9_]+(?:[.\-][a-z0-9_]+)*")


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def tokenize_es(text: str) -> list[str]:
    """Tokenizador para BM25 pensado para documentación técnica en español.

    - Minúsculas y sin tildes ("paginación" == "paginacion").
    - Conserva términos compuestos como tokens propios ("ap-4091",
      "client.refunds.list", "idempotency_key") y además sus partes
      ("ap", "4091", "refunds", "list"), así coincide tanto la búsqueda exacta
      como la parcial.
    - Quita stopwords.
    """
    text = _strip_accents(text.lower())
    tokens: list[str] = []
    for tok in TOKEN_RE.findall(text):
        tokens.append(tok)
        parts = re.split(r"[.\-]", tok)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class RAGSystem:
    """Encapsula el EnsembleRetriever (BM25 + Pinecone) y devuelve los top-k documentos."""

    def __init__(
        self,
        vectorstore: VectorStore,
        documents: list[Document],
        *,
        k: int = config.TOP_K,
        candidate_k: int = config.CANDIDATE_K,
        weights: tuple[float, float] = (config.BM25_WEIGHT, config.SEMANTIC_WEIGHT),
        llm=None,
    ) -> None:
        if not documents:
            raise ValueError("El corpus para BM25 está vacío: corré primero `python ingest.py`.")
        self.vectorstore = vectorstore
        self.documents = documents
        self.k = k
        self.candidate_k = max(candidate_k, k)
        self.weights = list(weights)
        self.llm = llm
        self._bm25_cache: dict[tuple[str | None, int], BM25Retriever] = {}

    # -- Construcción desde Pinecone -------------------------------------------------
    @classmethod
    def from_pinecone(cls, *, namespace: str = config.NAMESPACE, with_llm: bool = False) -> "RAGSystem":
        from langchain_pinecone import PineconeVectorStore

        index = config.get_pinecone().Index(config.INDEX_NAME)
        vectorstore = PineconeVectorStore(
            index=index, embedding=config.get_embeddings(), text_key="text", namespace=namespace
        )
        documents = load_corpus(index, namespace)
        return cls(vectorstore, documents, llm=config.get_llm() if with_llm else None)

    # -- Retrievers ---------------------------------------------------------------
    def _bm25(self, category: str | None, k: int) -> BM25Retriever:
        """BM25 sobre todo el corpus o sobre una categoría (equivalente al filtro duro)."""
        key = (category, k)
        if key not in self._bm25_cache:
            docs = [d for d in self.documents if category is None or d.metadata.get("category") == category]
            if not docs:
                raise ValueError(f"No hay documentos con categoría '{category}'.")
            self._bm25_cache[key] = BM25Retriever.from_documents(docs, preprocess_func=tokenize_es, k=k)
        return self._bm25_cache[key]

    def _semantic(self, category: str | None, k: int):
        search_kwargs: dict = {"k": k}
        if category:
            # Filtro duro de Pinecone: se aplica DENTRO de la búsqueda, antes del ranking.
            search_kwargs["filter"] = {"category": {"$eq": category}}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def ensemble(self, category: str | None = None, candidate_k: int | None = None) -> EnsembleRetriever:
        ck = candidate_k or self.candidate_k
        return EnsembleRetriever(
            retrievers=[self._bm25(category, ck), self._semantic(category, ck)],
            weights=self.weights,
            id_key="chunk_id",  # deduplica por ID de chunk, no por texto
        )

    # -- API pública ----------------------------------------------------------------
    def retrieve(self, query: str, *, mode: Mode = "hybrid", category: str | None = None,
                 candidate_k: int | None = None) -> list[Document]:
        """Devuelve los top-k documentos para la consulta.

        candidate_k solo aplica al modo híbrido: cuántos candidatos aporta cada
        recuperador antes de fusionar (por defecto CANDIDATE_K).
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("La consulta no puede estar vacía.")
        if mode == "hybrid":
            docs = self.ensemble(category, candidate_k).invoke(query)
        elif mode == "semantic":
            docs = self._semantic(category, self.k).invoke(query)
        elif mode == "bm25":
            docs = self._bm25(category, self.k).invoke(query)
        else:
            raise ValueError(f"Modo desconocido: {mode}")
        # Copias con el puesto en el ranking (no se modifican los docs del corpus BM25).
        return [
            Document(id=d.id, page_content=d.page_content, metadata={**d.metadata, "rank": rank})
            for rank, d in enumerate(docs[: self.k], start=1)
        ]

    def answer(self, query: str, *, category: str | None = None) -> tuple[str, list[Document]]:
        """(Opcional) Genera una respuesta con el LLM usando solo los documentos recuperados."""
        if self.llm is None:
            raise RuntimeError("RAGSystem se creó sin LLM (usá with_llm=True).")
        docs = self.retrieve(query, category=category)
        context = "\n\n---\n\n".join(f"[{d.metadata['chunk_id']}]\n{d.page_content}" for d in docs)
        prompt = (
            "Respondé en español usando SOLO el CONTEXTO. Si la respuesta no está, decí "
            "exactamente \"No lo sé\". Citá entre corchetes los chunk_id que usaste.\n\n"
            f"CONTEXTO:\n{context}\n\nPREGUNTA: {query}"
        )
        return self.llm.invoke(prompt).text, docs


def load_corpus(index, namespace: str = config.NAMESPACE, batch: int = 100) -> list[Document]:
    """Trae del índice todos los chunks del namespace (texto + metadatos) para BM25.

    El índice es la fuente de verdad: BM25 y la búsqueda vectorial trabajan
    exactamente sobre los mismos chunks, con los mismos IDs.
    """
    ids: list[str] = []
    for page in index.list(namespace=namespace):
        ids.extend(page)
    docs: list[Document] = []
    for i in range(0, len(ids), batch):
        res = index.fetch(ids=ids[i:i + batch], namespace=namespace)
        for vid, vec in res.vectors.items():
            meta = dict(vec.metadata or {})
            text = meta.pop("text", "")
            meta["chunk_id"] = meta.get("chunk_id", vid)
            docs.append(Document(id=vid, page_content=text, metadata=meta))
    return sorted(docs, key=lambda d: d.metadata["chunk_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta al recuperador híbrido")
    parser.add_argument("pregunta")
    parser.add_argument("--modo", choices=["hybrid", "semantic", "bm25"], default="hybrid")
    parser.add_argument("--categoria", default=None, help="filtro duro por categoría")
    parser.add_argument("--responder", action="store_true", help="generar respuesta con el LLM")
    args = parser.parse_args()

    rag = RAGSystem.from_pinecone(with_llm=args.responder)
    if args.responder:
        text, docs = rag.answer(args.pregunta, category=args.categoria)
    else:
        docs = rag.retrieve(args.pregunta, mode=args.modo, category=args.categoria)
    print(f"\nTop-{len(docs)} ({args.modo}) para: {args.pregunta}\n")
    for d in docs:
        m = d.metadata
        print(f"  {m['rank']}. {m['chunk_id']:<40} [{m['category']}] {m['source']} pág.{m['page']}")
    if args.responder:
        print(f"\nRespuesta:\n{text}")


if __name__ == "__main__":
    main()
