# PA Day 2 — Track B: RAG & Vector DBs
## Subtopic 1 — Chunking Strategies with Modern Open-Source APIs

This is a local-first RAG engineering lab. It replaces hand-written chunking and BM25 code with maintained open-source APIs while preserving the original learning objective: compare chunk boundaries and their effect on retrieval.

### Stack
- **LangChain Core / Text Splitters:** document objects, recursive character, token and Markdown-aware splitting.
- **Chonkie:** optional semantic chunking.
- **Sentence Transformers:** local embedding model `all-MiniLM-L6-v2`.
- **Qdrant Client:** local in-memory vector index.
- **rank-bm25:** lexical retrieval baseline.
- **Streamlit:** interactive comparison UI.

No proprietary API key is required. Ollama and LangGraph are deliberately reserved for the next agentic RAG sprint: this project first isolates chunking and retrieval quality before adding generation or agent decisions.

### Setup (Python 3.11 recommended)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run
```powershell
python main.py --size 300 --overlap 60 --retriever bm25
python main.py --size 300 --overlap 60 --retriever vector
python -m streamlit run app.py
```

### Strategies
1. Recursive character splitting with LangChain.
2. Token-based splitting with LangChain.
3. Markdown structure-aware splitting with heading metadata.
4. Semantic chunking with Chonkie (optional; the app displays the reason if it cannot initialize).

### Retrieval and evaluation
Compare BM25 with dense vector retrieval (Sentence Transformers + local Qdrant). Metrics:
- Answer coverage: complete reference answer exists in at least one chunk.
- Hit@1 and Hit@k: a retrieved chunk contains the complete reference answer.
- Average retrieved context characters.

The small built-in help-centre dataset and questions are a controlled learning example, not a production benchmark. Add realistic documents and paraphrased questions before choosing settings.

### Notes
- Qdrant uses `:memory:`; vectors are not persisted after the run.
- First dense retrieval downloads the Hugging Face embedding model.
- Chunk-size sliders represent characters for character splitters. Token splitting uses token units; the units are not equivalent.
- Chonkie semantic APIs and optional model dependencies can vary by release. If unavailable, other strategies continue to work and the app surfaces the error.
- Streamlit Cloud may have limited memory for transformer models. Use BM25 mode on small instances or deploy the vector mode locally.

### Next agentic RAG sprint
Add document loaders (PDF/HTML), persistent Qdrant, hybrid retrieval, reranking, Ollama generation with source citations, and a LangGraph agent that selects chunking/retrieval configuration based on document type and evaluation results.
