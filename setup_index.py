"""Inicialización de la infraestructura: crea el índice Serverless si no existe.

Uso:
    python setup_index.py

- Si el índice no existe, lo crea (Serverless, AWS us-east-1, métrica coseno,
  dimensión EMBEDDING_DIM) y espera a que esté listo.
- Si ya existe, verifica que la dimensión y la métrica coincidan con la
  configuración: si no, frena con un error claro en vez de fallar al subir datos.
"""

from __future__ import annotations

import logging
import time

import config

logger = logging.getLogger("rag.setup")


class IndexConfigError(RuntimeError):
    """El índice existe pero con una configuración incompatible."""


def ensure_index(
    pc,
    name: str = config.INDEX_NAME,
    dimension: int = config.EMBEDDING_DIM,
    metric: str = config.METRIC,
    cloud: str = config.PINECONE_CLOUD,
    region: str = config.PINECONE_REGION,
    timeout: float = 120.0,
    poll: float = 2.0,
):
    """Crea el índice si hace falta y devuelve el objeto Index ya listo para usar."""
    if pc.has_index(name):
        desc = pc.describe_index(name)
        if desc.dimension != dimension:
            raise IndexConfigError(
                f"El índice '{name}' tiene dimensión {desc.dimension}, pero los embeddings "
                f"configurados generan {dimension}. Cambiá INDEX_NAME o borrá el índice "
                f"(la dimensión de un índice no se puede modificar)."
            )
        if desc.metric != metric:
            raise IndexConfigError(
                f"El índice '{name}' usa la métrica '{desc.metric}' y se esperaba '{metric}'."
            )
        logger.info("El índice '%s' ya existe (dim=%d, %s).", name, dimension, metric)
    else:
        from pinecone import ServerlessSpec

        logger.info("Creando índice serverless '%s' (dim=%d, %s, %s/%s)...",
                    name, dimension, metric, cloud, region)
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    deadline = time.monotonic() + timeout
    while not pc.describe_index(name).status["ready"]:
        if time.monotonic() > deadline:
            raise TimeoutError(f"El índice '{name}' no quedó listo en {timeout:.0f} s.")
        time.sleep(poll)
    return pc.Index(name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    pc = config.get_pinecone()
    index = ensure_index(pc)
    stats = index.describe_index_stats()
    print(f"\nÍndice '{config.INDEX_NAME}' listo.")
    print(f"  dimensión: {stats.dimension} | métrica: {config.METRIC} | "
          f"{config.PINECONE_CLOUD}/{config.PINECONE_REGION}")
    print(f"  vectores totales: {stats.total_vector_count}")
    for ns, info in (stats.namespaces or {}).items():
        print(f"  namespace '{ns}': {info.vector_count} vectores")


if __name__ == "__main__":
    main()
