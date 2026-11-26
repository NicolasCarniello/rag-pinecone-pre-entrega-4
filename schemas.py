"""Esquema único de metadatos de cada chunk en Pinecone.

Todos los vectores se crean con `build_metadata()`. Tener UNA sola función que
arma y valida los metadatos evita el *schema drift* (campos con nombres o
formatos distintos según quién los cargó), que haría que un filtro como
{"category": {"$eq": "errores"}} se saltee documentos en silencio.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SourceType = Literal["markdown", "json", "pdf", "text"]


class ChunkMetadata(BaseModel):
    """Metadatos de un chunk. Solo tipos que Pinecone acepta (str, int, bool, list[str])."""

    text: str = Field(description="Texto original del chunk (se guarda para no consultar otra base)")
    chunk_id: str = Field(description="ID estable: <doc_id>::p<página>::c<n>")
    doc_id: str = Field(description="ID del documento de origen (nombre de archivo sin extensión)")
    source: str = Field(description="Archivo de origen dentro de /data")
    source_type: SourceType
    title: str
    category: str = Field(description="Etiqueta de categoría, útil para filtros duros")
    tags: list[str] = Field(default_factory=list)
    page: int = Field(ge=0, description="Página del PDF (desde 1). 0 = el documento no tiene páginas")
    section: str = ""
    chunk_index: int = Field(ge=0)
    n_tokens: int = Field(gt=0)
    char_count: int = Field(gt=0)
    ingested_at: str = Field(description="Fecha de ingesta en ISO 8601 (UTC)")

    @field_validator("category", "doc_id")
    @classmethod
    def lowercase_slug(cls, v: str) -> str:
        v = v.strip().lower()
        if not v:
            raise ValueError("no puede estar vacío")
        return v

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, v: list[str]) -> list[str]:
        return sorted({t.strip().lower() for t in v if t.strip()})


def build_metadata(
    *,
    text: str,
    doc_id: str,
    source: str,
    source_type: SourceType,
    title: str,
    category: str,
    tags: list[str],
    page: int,
    section: str,
    chunk_index: int,
    n_tokens: int,
    ingested_at: str | None = None,
) -> dict:
    """Única puerta de entrada para crear metadatos: valida y devuelve un dict listo para Pinecone."""
    meta = ChunkMetadata(
        text=text,
        chunk_id=f"{doc_id.lower()}::p{page}::c{chunk_index:02d}",
        doc_id=doc_id,
        source=source,
        source_type=source_type,
        title=title,
        category=category,
        tags=tags,
        page=page,
        section=section,
        chunk_index=chunk_index,
        n_tokens=n_tokens,
        char_count=len(text),
        ingested_at=ingested_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return meta.model_dump()
