"""Pipeline de ingesta: /data (Markdown, JSON, PDF) → chunks → embeddings → Pinecone.

Uso:
    python ingest.py              # sincroniza /data con el namespace configurado
    python ingest.py --reset      # vacía el namespace antes de cargar

Pasos:
1. Carga: cada formato tiene su loader y todos devuelven "secciones" con
   metadatos (título, categoría, etiquetas, página). Los PDF se cargan página
   por página para poder guardar el número de página.
2. Chunking: RecursiveCharacterTextSplitter medido en tokens (600, overlap 80),
   cortando primero por títulos y párrafos. A cada chunk se le antepone un
   encabezado con documento, sección y página, para que no pierda contexto.
3. Metadatos: se arman con `schemas.build_metadata()` (una sola función, validada
   con Pydantic) e incluyen el texto original, así la búsqueda devuelve el
   contenido sin consultar otra base.
4. Upsert por lotes de 100 en el namespace, con IDs deterministas: volver a
   correr la ingesta actualiza en lugar de duplicar. Los vectores que ya no
   corresponden a ningún chunk (archivo borrado o modificado) se eliminan.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from schemas import build_metadata

logger = logging.getLogger("rag.ingest")

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
HEADER_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # sin internet la primera vez
        warnings.warn(f"tiktoken no disponible ({exc}); se aproxima 1 token ≈ 4 caracteres.",
                      RuntimeWarning, stacklevel=2)
        return None


def count_tokens(text: str) -> int:
    enc = _encoder()
    return max(1, len(text) // 4) if enc is None else len(enc.encode(text))


# ---------------------------------------------------------------------------
# Loaders: cada uno devuelve secciones (Document) con metadatos del documento
# ---------------------------------------------------------------------------
@dataclass
class SourceDoc:
    doc_id: str
    source: str
    source_type: str
    title: str
    category: str
    tags: list[str]
    pages: list[tuple[int, str]]  # (número de página, texto); página 0 = sin páginas


def load_markdown(path: Path) -> SourceDoc:
    raw = path.read_text(encoding="utf-8")
    meta: dict = {}
    m = FRONTMATTER_RE.match(raw)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        raw = raw[m.end():]
    title_match = HEADER_RE.search(raw)
    return SourceDoc(
        doc_id=path.stem,
        source=path.name,
        source_type="markdown" if path.suffix == ".md" else "text",
        title=meta.get("title") or (title_match.group(2) if title_match else path.stem),
        category=meta.get("category", "general"),
        tags=list(meta.get("tags", [])),
        pages=[(0, raw.strip())],
    )


def load_json(path: Path) -> SourceDoc:
    """Convierte un JSON con `items` en texto Markdown (un bloque por item)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"# {data.get('title', path.stem)}", "", data.get("description", "")]
    for item in data.get("items", []):
        head = " · ".join(str(item[k]) for k in ("code", "exception") if k in item)
        if "http_status" in item:
            head += f" (HTTP {item['http_status']})"
        lines += ["", f"## {head}"]
        for key, label in (("description", "Descripción"), ("solution", "Solución")):
            if key in item:
                lines.append(f"{label}: {item[key]}")
    return SourceDoc(
        doc_id=path.stem,
        source=path.name,
        source_type="json",
        title=data.get("title", path.stem),
        category=data.get("category", "general"),
        tags=list(data.get("tags", [])),
        pages=[(0, "\n".join(lines).strip())],
    )


def load_pdf(path: Path) -> SourceDoc:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = re.sub(r"[ \t]+\n", "\n", page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    title = (reader.metadata.title if reader.metadata and reader.metadata.title else path.stem)
    return SourceDoc(
        doc_id=path.stem,
        source=path.name,
        source_type="pdf",
        title=title,
        category=path.stem.split("_")[0] if "_" in path.stem else path.stem,
        tags=["pdf"],
        pages=pages,
    )


LOADERS = {".md": load_markdown, ".txt": load_markdown, ".json": load_json, ".pdf": load_pdf}


def load_sources(data_dir: Path = config.DATA_DIR) -> list[SourceDoc]:
    data_dir = Path(data_dir)
    files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in LOADERS)
    if not files:
        raise FileNotFoundError(f"No hay documentos .md/.txt/.json/.pdf en {data_dir}")
    docs = [LOADERS[p.suffix.lower()](p) for p in files]
    ids = [d.doc_id for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Hay documentos con el mismo nombre base: {ids}")
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=count_tokens,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        keep_separator="start",
        strip_whitespace=True,
    )


def _sections_in(text: str, chunk: str, search_from: int) -> tuple[str, int]:
    """Secciones (##/###) que abarca el chunk, incluida la que estaba abierta al empezar."""
    start = text.find(chunk, search_from)
    if start == -1:
        start = search_from
    end = start + len(chunk)
    current, inside = "", []
    for m in HEADER_RE.finditer(text):
        if len(m.group(1)) < 2:
            continue
        if m.start() <= start:
            current = m.group(2)
        elif m.start() < end:
            inside.append(m.group(2))
    return " / ".join(([current] if current else []) + inside), start + 1


def chunk_sources(sources: list[SourceDoc], ingested_at: str | None = None) -> list[Document]:
    """Devuelve los chunks como Documents con metadatos ya validados."""
    splitter = build_splitter()
    ingested_at = ingested_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    chunks: list[Document] = []
    for src in sources:
        idx = 0
        for page, text in src.pages:
            search_from = 0
            for piece in splitter.split_text(text):
                section, search_from = _sections_in(text, piece, search_from)
                header = f"Documento: {src.title}"
                if page:
                    header += f" | Página {page}"
                if section:
                    header += f" | Sección: {section}"
                body = f"{header}\n\n{piece}"
                meta = build_metadata(
                    text=body, doc_id=src.doc_id, source=src.source,
                    source_type=src.source_type, title=src.title, category=src.category,
                    tags=src.tags, page=page, section=section, chunk_index=idx,
                    n_tokens=count_tokens(piece), ingested_at=ingested_at,
                )
                chunks.append(Document(id=meta["chunk_id"], page_content=body, metadata=meta))
                idx += 1
    return chunks


# ---------------------------------------------------------------------------
# Embeddings + upsert por lotes
# ---------------------------------------------------------------------------
def batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def list_ids(index, namespace: str) -> set[str]:
    ids: set[str] = set()
    for page in index.list(namespace=namespace):
        ids.update(page)
    return ids


def upsert_chunks(
    index,
    embeddings,
    chunks: list[Document],
    namespace: str = config.NAMESPACE,
    batch_size: int = config.UPSERT_BATCH_SIZE,
    dimension: int = config.EMBEDDING_DIM,
) -> int:
    """Embebe y sube los chunks en lotes. Devuelve la cantidad de llamadas de upsert."""
    calls = 0
    for batch in batched(chunks, batch_size):
        vectors = embeddings.embed_documents([d.page_content for d in batch])
        bad = {len(v) for v in vectors if len(v) != dimension}
        if bad:
            raise ValueError(
                f"Los embeddings tienen dimensión {bad} y el índice espera {dimension}. "
                "Revisá EMBEDDING_DIM / EMBEDDING_MODEL."
            )
        index.upsert(
            vectors=[
                {"id": d.metadata["chunk_id"], "values": v, "metadata": d.metadata}
                for d, v in zip(batch, vectors, strict=True)
            ],
            namespace=namespace,
        )
        calls += 1
        logger.info("Lote %d: %d vectores subidos.", calls, len(batch))
    return calls


def ingest(
    index,
    embeddings,
    *,
    data_dir: Path = config.DATA_DIR,
    namespace: str = config.NAMESPACE,
    reset: bool = False,
    wait_seconds: float = 30.0,
) -> dict:
    sources = load_sources(data_dir)
    chunks = chunk_sources(sources)
    new_ids = {c.metadata["chunk_id"] for c in chunks}

    if reset:
        try:
            index.delete(delete_all=True, namespace=namespace)
            logger.info("Namespace '%s' vaciado.", namespace)
        except Exception as exc:  # el namespace todavía no existe
            logger.info("No se vació el namespace (%s).", exc.__class__.__name__)

    calls = upsert_chunks(index, embeddings, chunks, namespace)

    # Sincronización: borrar vectores que ya no corresponden a ningún chunk.
    stale = sorted(list_ids(index, namespace) - new_ids) if not reset else []
    for batch in batched(stale, 1000):
        index.delete(ids=batch, namespace=namespace)
    if stale:
        logger.info("Eliminados %d vectores obsoletos.", len(stale))

    # Las estadísticas de Pinecone se actualizan con unos segundos de demora.
    deadline = time.monotonic() + wait_seconds
    while True:
        ns = (index.describe_index_stats().namespaces or {}).get(namespace)
        count = ns.vector_count if ns else 0
        if count == len(chunks) or time.monotonic() >= deadline:
            break
        time.sleep(2)

    return {
        "documentos": len(sources),
        "chunks": len(chunks),
        "llamadas_upsert": calls,
        "obsoletos_eliminados": len(stale),
        "vectores_en_namespace": count,
        "chunks_detalle": chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta de /data en Pinecone")
    parser.add_argument("--reset", action="store_true", help="vaciar el namespace antes de cargar")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from setup_index import ensure_index

    index = ensure_index(config.get_pinecone())
    result = ingest(index, config.get_embeddings(), reset=args.reset)

    print(f"\n{'chunk_id':<44}{'tipo':<10}{'categoría':<16}{'pág':>4}{'tokens':>8}")
    for c in result["chunks_detalle"]:
        m = c.metadata
        print(f"{m['chunk_id']:<44}{m['source_type']:<10}{m['category']:<16}{m['page']:>4}{m['n_tokens']:>8}")
    print(
        f"\n{result['documentos']} documentos → {result['chunks']} chunks, "
        f"{result['llamadas_upsert']} llamada(s) de upsert, "
        f"{result['obsoletos_eliminados']} obsoletos eliminados. "
        f"Namespace '{config.NAMESPACE}': {result['vectores_en_namespace']} vectores."
    )


if __name__ == "__main__":
    main()
