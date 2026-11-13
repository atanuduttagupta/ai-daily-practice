"""Modern open-source RAG chunking lab: LangChain + Chonkie + HF + Qdrant + BM25."""
import argparse, re
from dataclasses import dataclass
from typing import Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter, MarkdownHeaderTextSplitter
from rank_bm25 import BM25Okapi

DOC = """# Acme Drive Help Centre

## Plans and Billing
Acme Drive offers three plans. The Free plan includes 5 GB of storage and supports files up to 100 MB. The Plus plan costs 8 dollars per month and includes 500 GB of storage with a maximum file size of 5 GB. The Team plan costs 15 dollars per user per month and includes 2 TB of shared storage, admin controls and priority support.
Annual subscriptions receive a discount of 20 percent compared with monthly billing. If you cancel within 30 days of your first payment, you can request a full refund by contacting support. Failed payments are retried three times over seven days before the account is moved back to the Free plan.

## Sharing and Recovery
To share a file, open the file menu and choose Share, then enter an email address or create a link. Link expiry can be set from one hour to 90 days. Team admins can disable public links for the whole organisation.
Deleted files move to the Trash, where they remain for 30 days before being permanently removed. Previous versions are kept for 180 days on Plus and Team plans.

## Security and API
All files are encrypted with AES-256 at rest and TLS 1.3 in transit. Two-factor authentication is available on every plan and required for Team admins.
Each API key is limited to 600 requests per minute. Uploads larger than 100 MB must use the resumable upload endpoint, which splits the file into pieces of up to 16 MB. Webhooks notify applications when a file is created, changed or deleted.

## Support
Free plan emails are answered within two business days. Plus plan emails are answered within one business day, and Team customers receive priority responses within four hours.
"""
QA = [
("How much storage does Free include?", "The Free plan includes 5 GB of storage and supports files up to 100 MB."),
("What is the refund window?", "If you cancel within 30 days of your first payment, you can request a full refund by contacting support."),
("How long do deleted files stay in Trash?", "Deleted files move to the Trash, where they remain for 30 days before being permanently removed."),
("What is the API rate limit?", "Each API key is limited to 600 requests per minute."),
("How large are resumable upload pieces?", "Uploads larger than 100 MB must use the resumable upload endpoint, which splits the file into pieces of up to 16 MB."),
("What encryption protects files at rest?", "All files are encrypted with AES-256 at rest and TLS 1.3 in transit."),
]
@dataclass
class Chunk:
    strategy: str
    text: str
    metadata: dict[str, Any]

def make_chunks(size=300, overlap=60):
    doc = Document(page_content=DOC, metadata={"source": "acme-help-centre.md"})
    result = {}
    splitters = {
        "Recursive character": RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=min(overlap, size-1),
            separators=["\\n## ", "\\n\\n", ". ", " ", ""], add_start_index=True),
        "Token-based": TokenTextSplitter(
            chunk_size=max(50, size//4), chunk_overlap=max(0, overlap//4)),
    }
    for name, splitter in splitters.items():
        result[name] = [Chunk(name, d.page_content, dict(d.metadata))
                        for d in splitter.split_documents([doc])]
    md = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),("##","h2")], strip_headers=False)
    md_docs = md.split_text(DOC)
    bounded = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=min(overlap,size-1))
    result["Markdown structure-aware"] = [
        Chunk("Markdown structure-aware", d.page_content, {**d.metadata,"source":"acme-help-centre.md"})
        for d in bounded.split_documents(md_docs)]
    try:
        from chonkie import SemanticChunker
        sem = SemanticChunker(embedding_model="minishlab/potion-base-8M")
        cs = sem.chunk(DOC)
        result["Chonkie semantic"] = [
            Chunk("Chonkie semantic", getattr(c,"text",str(c)), {"source":"acme-help-centre.md"})
            for c in cs]
        result["_semantic_note"] = []
    except Exception as e:
        result["Chonkie semantic"] = []
        result["_semantic_note"] = [Chunk("note", str(e), {})]
    return result

def bm25_search(chunks, query, k=3):
    if not chunks: return []
    tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
    model = BM25Okapi([tok(c.text) for c in chunks])
    scores = model.get_scores(tok(query))
    order = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
    return [(chunks[i], float(scores[i])) for i in order[:k]]

def vector_search(chunks, query, k=3):
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient, models
    if not chunks: return []
    embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectors = embed.encode([c.text for c in chunks], normalize_embeddings=True).tolist()
    qv = embed.encode(query, normalize_embeddings=True).tolist()
    client = QdrantClient(":memory:")
    client.create_collection("chunk_lab", vectors_config=models.VectorParams(size=len(qv), distance=models.Distance.COSINE))
    client.upsert("chunk_lab", [models.PointStruct(id=i, vector=v, payload={"text":chunks[i].text,"metadata":chunks[i].metadata}) for i,v in enumerate(vectors)])
    hits = client.query_points("chunk_lab", query=qv, limit=min(k,len(chunks))).points
    return [(Chunk("vector result", h.payload["text"], h.payload.get("metadata",{})), float(h.score)) for h in hits]

def evaluate(chunks, retriever="bm25", k=3):
    if not chunks: return {"chunks":0,"coverage":0,"hit@1":0,"hit@k":0,"context_chars":0}
    hit1=hitk=covered=ctx=0
    for question,gold in QA:
        covered += any(gold in c.text for c in chunks)
        found = vector_search(chunks,question,k) if retriever=="vector" else bm25_search(chunks,question,k)
        hit1 += bool(found and gold in found[0][0].text)
        hitk += any(gold in c.text for c,_ in found)
        ctx += sum(len(c.text) for c,_ in found)
    n=len(QA)
    return {"chunks":len(chunks),"coverage":covered/n,"hit@1":hit1/n,"hit@k":hitk/n,"context_chars":ctx/n}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--size",type=int,default=300)
    ap.add_argument("--overlap",type=int,default=60)
    ap.add_argument("--retriever",choices=["bm25","vector"],default="bm25")
    a=ap.parse_args()
    for name,chunks in make_chunks(a.size,a.overlap).items():
        if name.startswith("_") or not chunks: continue
        print(name, evaluate(chunks,a.retriever))
if __name__=="__main__": main()
