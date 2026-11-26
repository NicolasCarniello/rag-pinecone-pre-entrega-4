# Pre-entrega 4 — Sistema RAG escalable en la nube con Pinecone

Módulo de recuperación en Python que lleva el RAG de la pre-entrega anterior (ChromaDB local) a la nube:

1. **Infraestructura**: crea (si no existe) un índice **Pinecone Serverless** y valida su configuración.
2. **Ingesta**: procesa documentación técnica en **Markdown, JSON y PDF**, la fragmenta y la sube por lotes con **metadatos avanzados** (fuente, página, categoría, etiquetas, sección, texto original).
3. **Recuperador híbrido**: la clase `RAGSystem` combina **BM25 (léxico)** y **Pinecone (semántico)** con un `EnsembleRetriever` y devuelve los **top-5** documentos.
4. **Evaluación**: `evaluate.py` mide **Recall@5** y **Precision@5** (más MRR) sobre un golden set y compara el híbrido contra cada recuperador por separado.

El dataset (`/data`) es la documentación de **`andespay-python`, una librería de Python ficticia** (SDK de pagos): guía de instalación, referencia de pagos, webhooks y reembolsos/paginación (Markdown), catálogo de códigos de error (JSON) y changelog de 3 páginas (PDF). Al ser ficticia, el modelo no puede responder "de memoria" y está llena de términos exactos (`AP-4290`, `PaymentIntent.confirm()`, `idempotency_key`), justo el caso donde la búsqueda léxica suma.

---

## Arquitectura

```
                    ┌────────────── INGESTA (ingest.py) ───────────────┐
 data/*.md ─┐       │ loaders por formato (PDF página por página)       │
 data/*.json├─────► │ → RecursiveCharacterTextSplitter (600 tok, 80 ov.)│
 data/*.pdf ┘       │ → build_metadata() validado con Pydantic          │
                    │ → embeddings gemini-embedding-001 (1536 dim)      │
                    │ → upsert por lotes de 100, IDs deterministas      │──► Pinecone Serverless
                    └───────────────────────────────────────────────────┘    índice "andespay-docs"
                                                                             namespace "dev"
                    ┌────────────── CONSULTA (rag_system.py) ──────────┐          │
 pregunta ────────► │ RAGSystem                                         │          │
                    │  ├─ BM25Retriever (corpus traído del índice) ─┐   │ ◄────────┘
                    │  └─ PineconeVectorStore (+ filtro opcional) ──┤   │
                    │     EnsembleRetriever pesos [0.4, 0.6], RRF ──┘   │
                    │     10 candidatos c/u → top-5                     │──► top-5 Documents
                    └───────────────────────────────────────────────────┘
```

## Estructura

| Archivo | Qué hace |
|---|---|
| `setup_index.py` | Crea el índice Serverless si no existe (AWS `us-east-1`, coseno, 1536 dim) y espera a que esté listo. Si existe con otra dimensión o métrica, frena con un error claro. |
| `ingest.py` | Pipeline de ingesta: carga, chunking, metadatos, embeddings y upsert por lotes. Sincroniza: elimina vectores obsoletos. |
| `schemas.py` | `ChunkMetadata` (Pydantic) y `build_metadata()`: la única función que arma metadatos, para evitar *schema drift*. |
| `rag_system.py` | Clase `RAGSystem` (EnsembleRetriever BM25 + Pinecone) y CLI de consulta. |
| `evaluate.py` | Recall@5, Precision@5 y MRR sobre `golden_set.json`; imprime el reporte, guarda `evidencia/metricas.json` y actualiza la sección de resultados de este README. |
| `golden_set.json` | 6 preguntas con su documento fuente esperado (la consigna pide 5). |
| `config.py` | Toda la configuración, leída de variables de entorno. |
| `tests/` | 26 tests offline con un Pinecone falso en memoria + 4 de integración contra Pinecone real. |
| `scripts/generar_changelog_pdf.py` | Regenera el PDF de ejemplo. |

---

## Cómo replicar el índice

### 1. Cuentas y claves (gratis)

- **Pinecone**: creá una cuenta en [app.pinecone.io](https://app.pinecone.io) (plan *Starter*, gratuito) → **API Keys** → *Create API key*.
- **Google AI Studio** (embeddings): [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → *Create API key*.

### 2. Entorno

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # completá PINECONE_API_KEY y GOOGLE_API_KEY
```

`.env` está en `.gitignore`; en el repo solo se versiona `.env.example`, sin valores.

| Variable | Valor por defecto | Para qué |
|---|---|---|
| `PINECONE_API_KEY` | — | Acceso a Pinecone |
| `GOOGLE_API_KEY` | — | Embeddings de Gemini |
| `INDEX_NAME` | `andespay-docs` | Nombre del índice |
| `PINECONE_NAMESPACE` | `dev` | Partición lógica (dev / staging / prod) |
| `EMBEDDING_DIM` | `1536` | Dimensión del índice **y** de los embeddings |

### 3. Crear el índice

```bash
python setup_index.py
```

Crea `andespay-docs` como **Serverless** en `aws/us-east-1` (la región del plan gratuito), con **métrica coseno** y **dimensión 1536**. Si el índice ya existe, verifica que la dimensión y la métrica coincidan con la configuración (la dimensión de un índice no se puede cambiar después).

### 4. Cargar los documentos

```bash
python ingest.py            # o --reset para vaciar el namespace antes
```

Imprime cada chunk con su tipo de fuente, categoría, página y tokens. Se puede correr las veces que haga falta: los IDs son deterministas (`<doc>::p<página>::c<n>`), así que actualiza en lugar de duplicar, y borra los vectores de chunks que ya no existen.

### 5. Consultar

```bash
python rag_system.py "¿Qué significa el código AP-4290?"
python rag_system.py "¿Cómo pagino los reembolsos?" --modo semantic
python rag_system.py "límite de peticiones" --categoria errores    # filtro duro
python rag_system.py "¿Cómo verifico un webhook?" --responder      # + respuesta del LLM
```

### 6. Evaluar

```bash
python evaluate.py
```

### 7. Tests

```bash
pytest -q                  # 26 offline (Pinecone falso) + 4 de integración si hay claves
```

---

## Decisiones de diseño

### Infraestructura (Pinecone)
- **Dimensión definida en un solo lugar** (`EMBEDDING_DIM`): se usa para crear el índice, para pedirle a Gemini embeddings de 1536 dimensiones (`output_dimensionality`) y para validar cada vector antes del upsert. Si algo no coincide, falla con un mensaje claro antes de subir nada.
- **Métrica coseno**, la recomendada para embeddings de texto.
- **Namespace por entorno** (`dev` por defecto): los datos de prueba no se mezclan con los de producción, y la búsqueda solo compite dentro del namespace.
- **Esperas explícitas**: `setup_index.py` espera a que el índice esté `ready`, y la ingesta espera a que las estadísticas reflejen los vectores subidos (Pinecone las actualiza con demora).

### Ingesta y metadatos
- **Un loader por formato**. Los PDF se leen **página por página**, así cada chunk sabe de qué página viene. El JSON de errores se convierte en un bloque por código. Los Markdown traen categoría y etiquetas en un *frontmatter* YAML.
- **Chunking**: `RecursiveCharacterTextSplitter` medido en **tokens** (600, overlap 80, dentro del rango recomendado de 500–800), cortando primero por títulos `##`/`###`, después por párrafos y líneas.
- **Encabezado de contexto** en cada chunk (`Documento | Página | Sección`), para que tanto el embedding como BM25 sepan de qué trata un fragmento aislado.
- **Metadatos** (todos con tipos que Pinecone acepta): `text` (texto original, así la búsqueda devuelve el contenido listo para el LLM sin consultar otra base), `chunk_id`, `doc_id`, `source`, `source_type`, `title`, `category`, `tags`, `page` (0 = documento sin páginas), `section`, `chunk_index`, `n_tokens`, `char_count` e `ingested_at` (ISO 8601).
- **Sin *schema drift***: todos los metadatos salen de `build_metadata()`, validada con Pydantic (normaliza la categoría y las etiquetas a minúsculas y rechaza tipos de fuente desconocidos).
- **Upsert por lotes de 100** e **IDs deterministas**: la ingesta es idempotente y sincroniza el índice con `/data`.

### Recuperador híbrido (`RAGSystem`)
- `BM25Retriever` + retriever de `PineconeVectorStore`, combinados con `EnsembleRetriever(weights=[0.4, 0.6], id_key="chunk_id")`, que fusiona con *Reciprocal Rank Fusion* ponderada y deduplica por ID de chunk.
- **El corpus de BM25 se trae del propio índice** (`index.list` + `index.fetch`): BM25 y la búsqueda vectorial trabajan sobre exactamente los mismos chunks.
- **Tokenizador de BM25 para documentación técnica en español**: minúsculas, sin tildes, sin stopwords, y conserva términos compuestos como un token y además sus partes (`ap-4290` → `ap-4290`, `ap`, `4290`; `client.refunds.list` → también `refunds`, `list`).
- **Filtro duro por categoría** (`--categoria`): en Pinecone se aplica dentro de la búsqueda (`{"category": {"$eq": ...}}`) y BM25 se arma solo con los chunks de esa categoría.
- **10 candidatos por recuperador → top-5**. Con 5 por lado (el esquema de la clase) y pesos 0.4/0.6, un resultado que **solo** encuentra BM25 nunca puede entrar al top-5: aunque sea el 1.º de BM25 suma `0.4/61 ≈ 0.0066`, menos que el 5.º semántico (`0.6/65 ≈ 0.0092`). BM25 solo reordenaría lo que ya trajo la búsqueda semántica. Con 10 candidatos por lado, un chunk que BM25 pone 1.º y la búsqueda semántica 8.º suma `0.4/61 + 0.6/68 ≈ 0.0154` y sube al primer lugar. Hay un test que lo demuestra, y la evaluación compara ambas configuraciones con datos reales.

### Evaluación
- Un chunk recuperado es **relevante** si pertenece al `documento_id_esperado`.
- **Recall@5** = documentos relevantes recuperados / documentos relevantes que existen. Como cada pregunta tiene un documento correcto, vale 1 si aparece entre los 5 y 0 si no.
- **Precision@5** = chunks relevantes entre los 5 / 5. **Tiene techo**: si el documento esperado tiene 2 chunks, lo máximo posible es 0,40. El reporte muestra ese máximo al lado, para no leer un 0,40 como un mal resultado.
- **MRR**: 1 / posición del primer chunk relevante (mide si el correcto aparece arriba).
- El golden set mezcla preguntas **léxicas** (códigos y nombres de métodos), **semánticas** (paráfrasis que no usan las palabras del documento) y **mixtas**, para que se note qué aporta cada recuperador.

---

## Resultados de la evaluación

<!-- METRICAS:INICIO -->
_Se completa automáticamente al correr `python evaluate.py`._
<!-- METRICAS:FIN -->
