import streamlit as st
import pandas as pd
from main import make_chunks, evaluate, bm25_search, vector_search, QA
st.set_page_config(page_title="RAG Chunking Lab",layout="wide")
st.title("RAG Chunking Strategies Lab")
st.caption("Modern open-source APIs: LangChain, Chonkie, Sentence Transformers and Qdrant.")
with st.sidebar:
    size=st.slider("Chunk size",150,900,300,50)
    overlap=st.slider("Overlap",0,min(250,size-1),min(60,size-1),10)
    retriever=st.selectbox("Retriever",["BM25","Dense vectors (Sentence Transformers + Qdrant)"])
    k=st.slider("Top-k",1,5,3)
chunks_by_strategy=make_chunks(size,overlap)
available={n:c for n,c in chunks_by_strategy.items() if not n.startswith("_") and c}
if retriever.startswith("Dense"):
    st.info("Dense retrieval downloads all-MiniLM-L6-v2 on first use.")
rows=[]
for name,chunks in available.items():
    with st.spinner(f"Evaluating {name}..."):
        m=evaluate(chunks,"vector" if retriever.startswith("Dense") else "bm25",k)
    rows.append({"Strategy":name,"Chunks":m["chunks"],"Answer coverage":f'{m["coverage"]:.0%}',"Hit@1":f'{m["hit@1"]:.0%}',f"Hit@{k}":f'{m["hit@k"]:.0%}',"Avg context chars":round(m["context_chars"])})
st.subheader("1. Strategy comparison")
st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
if chunks_by_strategy.get("_semantic_note"):
    with st.expander("Chonkie semantic chunker status"):
        st.warning(chunks_by_strategy["_semantic_note"][0].text)
st.subheader("2. Inspect chunk boundaries")
strategy=st.selectbox("Strategy",list(available))
cs=available[strategy]
idx=st.number_input("Chunk number",0,len(cs)-1,0)
st.text_area("Chunk text",cs[int(idx)].text,height=180)
st.json(cs[int(idx)].metadata)
st.subheader("3. Retrieval test")
question=st.selectbox("Question",[q for q,_ in QA])
if st.button("Retrieve",type="primary"):
    hits=vector_search(cs,question,k) if retriever.startswith("Dense") else bm25_search(cs,question,k)
    for rank,(chunk,score) in enumerate(hits,1):
        with st.expander(f"Rank {rank} · score {score:.4f}",expanded=rank==1):
            st.write(chunk.text)
            st.json(chunk.metadata)
st.markdown("**Metrics:** Answer coverage checks whether each gold answer is intact in any chunk. Hit@k checks whether a retrieved chunk contains it. This small curated set is for learning, not production decisions.")
