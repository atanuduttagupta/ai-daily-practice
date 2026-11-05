# Document Search Engine using TF-IDF

**Track:** Generative AI & Transformers --- Classical NLP\
**Subtopic:** Bag of Words and TF-IDF\
**Difficulty:** Beginner

## 1. Project Overview

This project demonstrates how classical Natural Language Processing
(NLP) techniques convert text documents into numerical representations
and use those representations for document retrieval.

It implements Bag of Words and TF-IDF manually, validates the
calculations against scikit-learn, and ranks documents using cosine
similarity.

The interactive Streamlit application lets users customize the document
corpus, enter search queries, configure stop-word removal, and inspect
the calculations.

## 2. Learning Objectives

-   Text tokenization and vocabulary construction.
-   Bag-of-Words representation and document-term matrices.
-   Term Frequency (TF) and Inverse Document Frequency (IDF).
-   TF-IDF weighting and normalization.
-   Stop-word removal.
-   Cosine similarity and document ranking.
-   Validation against scikit-learn.

## 3. Understanding TF-IDF

TF-IDF stands for **Term Frequency-Inverse Document Frequency**. It
measures how informative a word is within a document relative to the
entire corpus.

### 3.1 Term Frequency

This project uses raw term counts:

`TF(t,d) = count of term t in document d`

### 3.2 Inverse Document Frequency

The implementation uses scikit-learn-compatible smoothed IDF:

`IDF(t) = ln((1 + N) / (1 + DF(t))) + 1`

Where: - `N` is the total number of documents. - `DF(t)` is the number
of documents containing term `t`.

A term appearing in many documents receives a lower IDF than a rare
term. With this smoothed formula, a term appearing in every document
receives an IDF of 1, not 0.

### 3.3 TF-IDF

`TF-IDF(t,d) = TF(t,d) × IDF(t)`

The resulting document vectors are L2-normalized.

turns a tiny corpus into numbers the pre-transformer way.
Builds bag-of-words and TF-IDF by hand, verifies both against scikit-learn,
then ranks documents for a query using cosine similarity on the TF-IDF
vectors — a miniature search engine.
Term Frequency-Inverse Document Frequency
TF-IDF (Term Frequency-Inverse Document Frequency) is a text vectorization technique that converts text into numerical features. It measures how important a word is to a specific document relative to a whole collection of documents (corpus).
Unlike simple word counting (Bag of Words), TF-IDF penalizes universally common words so that unique, informative words stand out.

The Core Formula Breakdown
TF-IDF is calculated by multiplying two separate metrics:
TF-IDF = TF X IDF

• TF (Term Frequency): Measures how frequently a term appears in a single document.
TF = count of term in document / total terms in document
• Interview Insight: High TF means the word is highly relevant to this specific page.
• IDF (Inverse Document Frequency): Measures how rare or common a word is across the entire corpus.
IDF = log (Total number of documents / Number of documents containing this term )
• Interview Insight: If a word appears in every single document (like "the", "is", "run"), the denominator is high, making the IDF close to 0. This effectively filters out stop words naturally.

Pros:
• Simple & Fast: Computationally inexpensive compared to deep learning models.
• Filters Noise: Automatically down-weights common stop words without needing an explicit stop-word list.
• Good Baseline: Highly effective for basic text classification, search engine retrieval, and keyword extraction.
Cons:
• No Context/Semantics: It treats words as independent tokens. It cannot understand synonyms (e.g., "huge" vs. "large") or word order ("not good" vs. "good, not").
• Out-of-Vocabulary (OOV) Issue: It cannot handle new words that weren't present in the training corpus.
• Sparsity: The resulting vectors are large and mostly filled with zeros, which requires extra memory management.
Workout example:
Corpus Size = 1 Million
Word: Mathematics and The
TF

A History of Math Essay (100 words total)
Imagine a short academic paragraph. Because it is about math, the word "mathematics" appears 5 times. Because it is written in English, the stop word "the" appears 8 times.
• TF for "the":
TF = 8/100 = .08
• TF for "mathematics":
TF = 5/100 = .05

IDF
The Word: "Mathematics" (Rare / Specific Keyword)

"Mathematics" is a specific technical term. It will not show up in cooking recipes, sports recaps, or celebrity gossip—only in academic papers, school articles, or textbook chapters. Let’s assume it appears in 1,000 out of the 1,000,000 documents.
• Total Documents ((N)): 1,000,000
• Documents with term ((DF)): 1,000
IDF = log(1 Million / 1000) = log (1000) = 3
"The": It appears in all documents
IDF = log(1 Million / 1 Million ) = log ( 1 ) = 0
• Final Score for "mathematics": .05 * 3 = .15
• Final Score for "the": .08 * 0 = 0
Today's World:
While early RAG systems relied purely on AI vector embeddings (dense semantic search), production systems found that vector search often misses the mark on specific keywords, strict IDs, and codes. Today, TF-IDF / BM25 is deployed alongside vector search to achieve the best possible retrieval accuracy.
User Search Query
A. Dense Vector Index (Semantic Match) using BERT or open AI embedding
B. Sparse TF-IDF/BM25 (Keyword Match)


## 4. Stop-Word Removal

Common words such as "the", "is", and "and" may contribute little to
document retrieval. TF-IDF down-weights common terms, but does not
automatically eliminate every stop word.

The application provides two configurations:

-   **OFF:** retain all recognized tokens.
-   **ON:** remove common English stop words before vocabulary
    construction.

The same filtering is applied to the document corpus and search query.
Users can compare vocabulary sizes and similarity scores with both
settings.

## 5. Cosine Similarity and Document Search

`cosine_similarity(A,B) = (A · B) / (||A|| × ||B||)`

The query is transformed using the vocabulary and IDF values learned
from the document corpus. The normalized query vector is compared with
normalized document vectors, and documents are ranked by descending
similarity.

## 6. Project Architecture

  File                 Responsibility
  -------------------- ------------------------------------------------------
  `main.py`            Manual BoW, TF-IDF, cosine similarity and validation
  `app.py`             Interactive Streamlit application
  `project.yaml`       Structured AI Project Atlas metadata
  `ai-context.md`      Detailed knowledge for the future AI assistant
  `README.md`          Project documentation
  `requirements.txt`   Python dependencies

The core implementation includes `tokenize()`, `build_vocab()`,
`bag_of_words()`, `calculate_idf()`, `tfidf()`, `cosine_scores()`,
`search_documents()`, and `validate_against_sklearn()`.

The Streamlit application provides an editable corpus and query,
configurable stop-word removal, Top-K results, similarity scores,
vocabulary inspection, Bag-of-Words matrix, IDF values, TF-IDF matrix,
and scikit-learn validation.

## 7. Scikit-learn Validation

The manual implementation is verified against: - `CountVectorizer` -
`TfidfVectorizer`

Validation checks vocabulary equality, Bag-of-Words matrix equality, and
TF-IDF numerical equivalence. Both stop-word configurations are
validated.

## 8. Advantages and Limitations

### Advantages

-   Simple and computationally inexpensive.
-   Interpretable numerical representation of text.
-   Useful for basic search and information retrieval.
-   A strong foundation for understanding sparse retrieval.

### Limitations

-   No contextual or semantic understanding.
-   Does not recognize synonyms or word order.
-   Treats forms such as "market" and "markets" as different tokens.
-   Query terms outside the corpus vocabulary are ignored.
-   Can produce large, sparse document-term matrices.

## 9. Connection to Modern AI and RAG

TF-IDF and BM25 provide sparse keyword-based retrieval, while dense
vector search uses embeddings to capture semantic relationships.

Modern RAG systems can combine these approaches through hybrid
retrieval:

-   **Sparse retrieval:** useful for exact keywords, technical terms,
    identifiers and codes.
-   **Dense retrieval:** useful for semantic similarity and contextual
    relationships.
-   **Hybrid retrieval:** combines lexical and semantic retrieval to
    improve document discovery.

This project establishes the foundation for understanding sparse
retrieval before progressing to BM25, embeddings and hybrid RAG systems.

## 10. Installation

Create and activate a virtual environment:

``` powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

``` powershell
python -m pip install -r requirements.txt
```

## 11. Running the Project

### Command-line demonstration

``` powershell
python main.py
```

The demonstration runs with stop-word removal disabled and enabled, and
validates the manual calculations against scikit-learn.

### Interactive Streamlit application

``` powershell
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser. Enter your own documents
and query to experiment with TF-IDF document retrieval.

## 12. Future Enhancements

-   Stemming and lemmatization.
-   N-gram-based TF-IDF.
-   BM25 retrieval.
-   Sparse matrix optimization.
-   Semantic search using transformer embeddings.
-   Hybrid retrieval combining TF-IDF and dense embeddings.
-   Comparison of lexical and semantic retrieval.
-   Integration with RAG pipelines.

These are future enhancements and are not claimed as implemented
features.

## 13. Key Learning

TF-IDF provides an interpretable way to represent documents numerically
and rank them according to lexical similarity. However, it does not
understand meaning or context. This gap motivates embeddings,
transformers and hybrid retrieval techniques in modern AI systems.
