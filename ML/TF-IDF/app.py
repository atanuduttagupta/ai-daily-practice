
"""Streamlit application for the TF-IDF Document Search Explorer."""

import streamlit as st

from main import search_documents, validate_against_sklearn


DEFAULT_DOCUMENTS = """the cat sat on the mat
the dog chased the cat
dogs and cats are popular pets
stock markets fell sharply on monday
investors watch the stock market closely
"""

st.set_page_config(
    page_title="TF-IDF Document Search Explorer",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 TF-IDF Document Search Explorer")

st.markdown(
    """
    Explore how Bag of Words and TF-IDF convert documents into numerical
    representations, and discover how search engines rank documents using
    cosine similarity.
    """
)

st.divider()

# --------------------------------------------------
# 1. User inputs
# --------------------------------------------------

st.subheader("1. Configure your document search")

col1, col2 = st.columns(2)

with col1:
    corpus_text = st.text_area(
        "Document Corpus",
        value=DEFAULT_DOCUMENTS,
        height=250,
        help="Enter one document per line.",
    )

with col2:
    query = st.text_area(
        "Search Query",
        value="stock market news",
        height=120,
        help="Enter the text you want to search for.",
    )

    remove_stop_words = st.checkbox(
        "Remove English stop words",
        value=True,
        help="Removes common words such as 'the', 'is', 'and', etc.",
    )

    top_k = st.number_input(
        "Number of search results",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

search_clicked = st.button(
    "Search Documents",
    type="primary",
    use_container_width=True,
)

# --------------------------------------------------
# 2. Process corpus and search
# --------------------------------------------------

if search_clicked:

    docs = [
        line.strip()
        for line in corpus_text.splitlines()
        if line.strip()
    ]

    if not docs:
        st.error("Please enter at least one document.")

    elif not query.strip():
        st.error("Please enter a search query.")

    else:
        try:
            result = search_documents(
                docs=docs,
                query=query,
                remove_stop_words=remove_stop_words,
            )

            validation_passed = validate_against_sklearn(
                docs=docs,
                remove_stop_words=remove_stop_words,
            )

            st.session_state["search_result"] = result
            st.session_state["documents"] = docs
            st.session_state["query"] = query
            st.session_state["top_k"] = int(top_k)
            st.session_state["validation_passed"] = validation_passed

        except ValueError as error:
            st.error(str(error))

        except AssertionError:
            st.error(
                "Manual TF-IDF and scikit-learn validation failed. "
                "Please check the corpus and implementation."
            )

# --------------------------------------------------
# 3. Display results
# --------------------------------------------------

if "search_result" in st.session_state:

    result = st.session_state["search_result"]
    docs = st.session_state["documents"]

    st.divider()
    st.subheader("2. Search Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Documents", len(docs))
    col2.metric("Vocabulary Size", len(result["vocabulary"]))
    col3.metric(
        "Vocabulary Features",
        f"{result['counts'].shape[0]} × {result['counts'].shape[1]}",
    )

    if st.session_state["validation_passed"]:
        st.success(
            "Manual Bag-of-Words and TF-IDF calculations "
            "match scikit-learn."
        )

    query_tokens = st.session_state["query"]

    st.markdown(f'**Query:** "{query_tokens}"')

    ranked_indices = result["ranked_indices"]
    scores = result["scores"]

    top_k_value = min(
        st.session_state["top_k"],
        len(docs),
    )

    for rank, index in enumerate(
        ranked_indices[:top_k_value],
        start=1,
    ):
        with st.container(border=True):
            col1, col2 = st.columns([5, 1])

            col1.markdown(f"**{rank}. {docs[index]}**")

            col2.metric(
                "Similarity",
                f"{scores[index]:.3f}",
            )

            st.progress(
                float(max(0.0, min(1.0, scores[index])))
            )

    # --------------------------------------------------
    # 4. Inspect calculations
    # --------------------------------------------------

    st.divider()
    st.subheader("3. Explore TF-IDF Calculations")

    tab1, tab2, tab3 = st.tabs(
        [
            "Vocabulary & Bag of Words",
            "IDF Values",
            "TF-IDF Matrix",
        ]
    )

    vocabulary = result["vocabulary"]
    counts = result["counts"]
    idf = result["idf"]
    weights = result["tfidf"]

    with tab1:
        st.markdown("**Document-Term Matrix (Bag of Words)**")

        bow_data = {
            "Document": [
                f"Document {i}"
                for i in range(len(docs))
            ]
        }

        for word_index, word in enumerate(vocabulary):
            bow_data[word] = counts[:, word_index].astype(int).tolist()

        st.dataframe(
            bow_data,
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        st.markdown(
            r"**Smoothed IDF: "
            r"$\ln((1+N)/(1+DF)) + 1$**"
        )

        idf_data = {
            "Word": vocabulary,
            "Document Frequency": (
                (counts > 0).sum(axis=0).astype(int).tolist()
            ),
            "IDF": idf.tolist(),
        }

        st.dataframe(
            idf_data,
            use_container_width=True,
            hide_index=True,
        )

    with tab3:
        st.markdown("**L2-normalized TF-IDF Matrix**")

        tfidf_data = {
            "Document": [
                f"Document {i}"
                for i in range(len(docs))
            ]
        }

        for word_index, word in enumerate(vocabulary):
            tfidf_data[word] = weights[:, word_index].tolist()

        st.dataframe(
            tfidf_data,
            use_container_width=True,
            hide_index=True,
        )

st.divider()

st.caption(
    "AI Project Atlas | Machine Learning | "
    "Classical NLP — Bag of Words & TF-IDF"
)
