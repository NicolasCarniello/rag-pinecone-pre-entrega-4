"""Evaluación cuantitativa del recuperador: Recall@5 y Precision@5 sobre un golden set.

Uso:
    python evaluate.py              # evalúa, imprime el reporte y actualiza el README
    python evaluate.py --no-readme  # solo imprime y guarda evidencia/metricas.json

Definiciones (con k = 5):
- Un chunk recuperado es RELEVANTE si pertenece al documento esperado
  (`documento_id_esperado` del golden set).
- Recall@k   = documentos relevantes recuperados / documentos relevantes existentes.
               Cada pregunta tiene UN documento correcto, así que vale 1 si ese
               documento aparece entre los k chunks recuperados y 0 si no.
- Precision@k = chunks relevantes entre los k recuperados / k.
               Su techo depende de cuántos chunks tiene el documento esperado:
               si tiene 2 chunks, lo máximo posible es 2/5 = 0,4.
- Además se informa MRR (1 / posición del primer chunk relevante) y
  Recall@k por chunks (chunks relevantes recuperados / chunks del documento).

Se comparan cuatro configuraciones: híbrido (10 candidatos por recuperador → top-5),
híbrido con 5 candidatos por recuperador (el esquema de la clase), solo semántico y solo BM25.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean

import config

K = 5
# (clave, modo de RAGSystem, candidatos por recuperador, nombre)
MODES = [
    ("hybrid", "hybrid", None, f"Híbrido ({config.CANDIDATE_K} cand. c/u → top-5)"),
    ("hybrid_k5", "hybrid", 5, "Híbrido (5 cand. c/u, como en clase)"),
    ("semantic", "semantic", None, "Solo semántico (Pinecone)"),
    ("bm25", "bm25", None, "Solo léxico (BM25)"),
]
EVIDENCE = config.BASE_DIR / "evidencia" / "metricas.json"
README = config.BASE_DIR / "README.md"
README_START, README_END = "<!-- METRICAS:INICIO -->", "<!-- METRICAS:FIN -->"


# ---------------------------------------------------------------------------
# Métricas (funciones puras)
# ---------------------------------------------------------------------------
def precision_at_k(retrieved: list[str], relevant: set[str], k: int = K) -> float:
    """Fracción de los k recuperados que son relevantes."""
    return sum(1 for r in retrieved[:k] if r in relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int = K) -> float:
    """Fracción de los relevantes existentes que aparecen entre los k recuperados."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str], k: int = K) -> float:
    for i, r in enumerate(retrieved[:k], start=1):
        if r in relevant:
            return 1 / i
    return 0.0


def evaluate_question(retrieved_docs: list[str], retrieved_chunks: list[str],
                      expected_doc: str, doc_chunks: set[str], k: int = K) -> dict:
    """Métricas de una pregunta.

    retrieved_docs: doc_id de cada chunk recuperado, en orden.
    retrieved_chunks: chunk_id de cada chunk recuperado, en orden.
    doc_chunks: todos los chunk_id del documento esperado (para el techo y el recall por chunks).
    """
    return {
        "recall@k": recall_at_k(list(dict.fromkeys(retrieved_docs)), {expected_doc}, k),
        "precision@k": precision_at_k(retrieved_docs, {expected_doc}, k),
        "precision_max": min(len(doc_chunks), k) / k,
        "recall_chunks@k": recall_at_k(retrieved_chunks, doc_chunks, k),
        "mrr": reciprocal_rank(retrieved_docs, {expected_doc}, k),
    }


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
def load_golden_set(path: Path = config.GOLDEN_SET) -> list[dict]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    for it in items:
        if not {"pregunta", "documento_id_esperado"} <= it.keys():
            raise ValueError(f"Item inválido en el golden set: {it}")
    return items


def run(rag, golden: list[dict], k: int = K) -> dict:
    chunks_by_doc: dict[str, set[str]] = {}
    for d in rag.documents:
        chunks_by_doc.setdefault(d.metadata["doc_id"], set()).add(d.metadata["chunk_id"])
    missing = {g["documento_id_esperado"] for g in golden} - chunks_by_doc.keys()
    if missing:
        raise ValueError(f"El golden set espera documentos que no están en el índice: {missing}")

    report: dict = {"modos": {}}
    for key, mode, cand, label in MODES:
        rows = []
        for g in golden:
            docs = rag.retrieve(g["pregunta"], mode=mode, candidate_k=cand) if mode == "hybrid" \
                else rag.retrieve(g["pregunta"], mode=mode)
            expected = g["documento_id_esperado"]
            m = evaluate_question(
                [d.metadata["doc_id"] for d in docs], [d.metadata["chunk_id"] for d in docs],
                expected, chunks_by_doc[expected], k,
            )
            rows.append({**g, **m, "recuperados": [d.metadata["chunk_id"] for d in docs]})
        report["modos"][key] = {
            "nombre": label,
            "preguntas": rows,
            "promedios": {key: round(mean(r[key] for r in rows), 3)
                          for key in ("recall@k", "precision@k", "precision_max", "recall_chunks@k", "mrr")},
        }
    return report


def format_report(report: dict, k: int = K) -> str:
    out = []
    hybrid = report["modos"]["hybrid"]
    out.append(f"Detalle por pregunta — {hybrid['nombre']}\n")
    out.append(f"{'#':<3}{'Esperado':<30}{'R@'+str(k):>6}{'P@'+str(k):>7}{'Máx P':>7}{'MRR':>6}  Pregunta")
    for i, r in enumerate(hybrid["preguntas"], 1):
        out.append(f"{i:<3}{r['documento_id_esperado']:<30}{r['recall@k']:>6.2f}{r['precision@k']:>7.2f}"
                   f"{r['precision_max']:>7.2f}{r['mrr']:>6.2f}  {r['pregunta'][:60]}")
    out.append(f"\nResumen ({len(hybrid['preguntas'])} preguntas, k={k})\n")
    out.append(f"{'Modo':<36}{'Recall@'+str(k):>10}{'Precision@'+str(k):>13}{'(máx.)':>8}{'MRR':>7}")
    for mode, data in report["modos"].items():
        p = data["promedios"]
        out.append(f"{data['nombre']:<36}{p['recall@k']:>10.2f}{p['precision@k']:>13.2f}"
                   f"{p['precision_max']:>8.2f}{p['mrr']:>7.2f}")
    return "\n".join(out)


def markdown_summary(report: dict, k: int = K) -> str:
    lines = [
        f"_Última evaluación: {report['fecha']} · índice `{report['indice']}` · namespace "
        f"`{report['namespace']}` · embeddings `{report['embeddings']}` · {report['n_chunks']} chunks._",
        "",
        f"| Modo | Recall@{k} | Precision@{k} | Precision@{k} máxima posible | MRR |",
        "|---|---|---|---|---|",
    ]
    for data in report["modos"].values():
        p = data["promedios"]
        lines.append(f"| {data['nombre']} | {p['recall@k']:.2f} | {p['precision@k']:.2f} | "
                     f"{p['precision_max']:.2f} | {p['mrr']:.2f} |")
    lines += ["", "Detalle del modo híbrido:", "",
              f"| # | Pregunta | Documento esperado | Recall@{k} | Precision@{k} | MRR |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(report["modos"]["hybrid"]["preguntas"], 1):
        lines.append(f"| {i} | {r['pregunta']} | `{r['documento_id_esperado']}` | {r['recall@k']:.2f} | "
                     f"{r['precision@k']:.2f} | {r['mrr']:.2f} |")
    return "\n".join(lines)


def update_readme(summary: str, readme: Path = README) -> bool:
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(text):
        return False
    readme.write_text(pattern.sub(lambda _: f"{README_START}\n{summary}\n{README_END}", text), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación Recall@5 / Precision@5")
    parser.add_argument("--no-readme", action="store_true", help="no actualizar el README")
    args = parser.parse_args()

    from rag_system import RAGSystem

    rag = RAGSystem.from_pinecone()
    golden = load_golden_set()
    report = run(rag, golden)
    report.update({
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "indice": config.INDEX_NAME,
        "namespace": config.NAMESPACE,
        "embeddings": f"{config.EMBEDDING_PROVIDER}:{config.EMBEDDING_MODEL} ({config.EMBEDDING_DIM} dim)",
        "n_chunks": len(rag.documents),
        "k": K,
    })

    print(format_report(report))
    EVIDENCE.parent.mkdir(exist_ok=True)
    EVIDENCE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetalle guardado en {EVIDENCE.relative_to(config.BASE_DIR)}")
    if not args.no_readme and update_readme(markdown_summary(report)):
        print("Resumen actualizado en README.md")


if __name__ == "__main__":
    main()
