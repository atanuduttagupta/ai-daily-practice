
# AI Context — Document Search Engine using TF-IDF

## 1. Project Identity

**Project title:** Document Search Engine using TF-IDF

**Domain:** Machine Learning

**Subdomain:** Natural Language Processing (Classical NLP)

**Difficulty:** Beginner

**Project type:** Machine Learning / Information Retrieval

**Source repository:** https://github.com/atanuduttagupta/ai-daily-practice

**Project directory:** `ML/TF-IDF/`

**Source folder URL:** https://github.com/atanuduttagupta/ai-daily-practice/tree/main/ML/TF-IDF

**Interactive application:** Streamlit-based TF-IDF Document Search Explorer

## 2. Project Overview

This project demonstrates how classical NLP techniques convert text documents into numerical representations and use those representations for document retrieval.

It implements Bag of Words and TF-IDF manually, validates the results against scikit-learn, and ranks documents according to cosine similarity with a user-provided search query.

The Streamlit application allows users to customize the document corpus, enter search queries, configure stop-word removal, and inspect intermediate calculations.

The project demonstrates the foundations of lexical information retrieval used in search engines and modern hybrid retrieval systems.

## 3. Learning Objectives

A learner completing this project should understand:

1. Text tokenization and vocabulary construction.
2. Bag-of-Words representation.
3. Document-term matrices.
4. Term Frequency (TF).
5. Inverse Document Frequency (IDF).
6. TF-IDF weighting and normalization.
7. Stop-word removal.
8. Cosine similarity.
9. Document ranking and information retrieval.
10. Validation of manually implemented algorithms against scikit-learn.

## 4. Core Concepts and Mathematics

### 4.1 Tokenization

Tokenization converts text into individual terms.

The implementation:
- Converts text to lowercase.
- Uses a regular expression to identify tokens.
- Retains tokens containing at least two word characters.
- Optionally removes English stop words.

Example:

Input:
`The cat sat on the mat`

Without stop-word removal:
`["the", "cat", "sat", "on", "the", "mat"]`

With stop-word removal:
`["cat", "sat", "mat"]`

### 4.2 Vocabulary

The vocabulary is the sorted collection of unique tokens found across all documents.

Each vocabulary term is assigned a column in the document-term matrix.

### 4.3 Bag of Words

Bag of Words represents each document using token counts.

Rows represent documents.
Columns represent vocabulary terms.
Values represent the number of occurrences of each term.

Word order is discarded.

### 4.4 Term Frequency

Term Frequency measures how frequently a term occurs in a document.

The implementation uses raw term counts as its term-frequency values.

### 4.5 Inverse Document Frequency

IDF measures how informative a term is across the document corpus.

The implementation uses scikit-learn-compatible smoothed IDF:

`IDF(t) = ln((1 + N) / (1 + DF(t))) + 1`

Where:

- N = number of documents.
- DF(t) = number of documents containing term t.

A term appearing in many documents receives a lower IDF than a rare term.

With this smoothed formula, even a term appearing in every document receives an IDF of 1.

### 4.6 TF-IDF

TF-IDF combines term frequency and inverse document frequency.

`TF-IDF(t,d) = TF(t,d) × IDF(t)`

The resulting document vectors are L2-normalized.

### 4.7 Stop-Word Removal

Common words such as "the", "is", and "and" may contribute little to document retrieval.

The application provides configurable English stop-word removal.

Two modes are supported:

- OFF: retain all recognized tokens.
- ON: remove English stop words before vocabulary construction and vectorization.

Stop-word removal is applied consistently to the document corpus and search query.

The application allows learners to compare both configurations.

### 4.8 Cosine Similarity

Cosine similarity measures the similarity between two vectors.

`cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)`

The query is transformed using the vocabulary and IDF values learned from the document corpus.

The normalized query vector is compared with the normalized document vectors using dot products.

Documents are ranked in descending order of similarity.

An empty or out-of-vocabulary query produces zero similarity scores rather than a division-by-zero error.

## 5. Implementation Architecture

### main.py — Core implementation

Contains reusable functions:

- `tokenize()`
- `build_vocab()`
- `bag_of_words()`
- `calculate_idf()`
- `tfidf()`
- `cosine_scores()`
- `search_documents()`
- `validate_against_sklearn()`

Responsibilities:

- Manual Bag-of-Words implementation.
- Manual TF-IDF calculation.
- Configurable stop-word removal.
- Query vectorization using corpus IDF.
- Cosine similarity calculation.
- Document ranking.
- Validation against scikit-learn.

### app.py — Streamlit application

Provides the interactive interface.

Features:

- Customizable document corpus.
- User-defined search query.
- Configurable stop-word removal.
- Top-K search results.
- Cosine similarity scores.
- Vocabulary inspection.
- Bag-of-Words matrix.
- IDF values.
- TF-IDF matrix.
- Manual implementation validation status.

The application imports reusable functions from `main.py` rather than duplicating the algorithms.

### project.yaml

Contains structured project metadata for future AI Project Atlas integration.

### README.md

Provides learner-facing explanations, setup instructions, and execution guidance.

### requirements.txt

Lists Python dependencies.

## 6. Scikit-learn Validation

The project verifies its manual implementation against:

- `CountVectorizer`
- `TfidfVectorizer`

Validation checks:

1. Vocabulary equality.
2. Bag-of-Words matrix equality.
3. TF-IDF matrix numerical equivalence.

The manual implementation uses the same smoothed IDF formula and L2 normalization as scikit-learn.

This validation is performed with stop-word removal both enabled and disabled.

## 7. Example Demonstration

Default document corpus:

1. the cat sat on the mat
2. the dog chased the cat
3. dogs and cats are popular pets
4. stock markets fell sharply on monday
5. investors watch the stock market closely

Example query:

`stock market news`

Expected behaviour:

- Documents containing matching query terms receive higher similarity scores.
- Documents without matching terms receive zero similarity.
- Stop-word removal changes the vocabulary and may change ranking scores.
- "market" and "markets" are treated as separate tokens.

The Streamlit application allows users to replace the example corpus and query.

## 8. Limitations

This project demonstrates lexical retrieval, not semantic understanding.

Known limitations:

1. Word order is discarded by Bag of Words.
2. Synonyms are not recognized.
3. Different word forms such as "market" and "markets" are treated as different tokens.
4. Out-of-vocabulary query terms are ignored.
5. The vocabulary is limited to the supplied document corpus.
6. Sparse text representations can become large for extensive corpora.
7. Stop-word removal may remove meaningful terms in certain contexts.

## 9. Future Enhancements

Potential extensions:

- Stemming and lemmatization.
- N-gram-based TF-IDF.
- BM25 retrieval.
- Sparse matrix optimization.
- Semantic search using transformer embeddings.
- Hybrid retrieval combining TF-IDF and dense vector search.
- Comparison of lexical and semantic retrieval.
- Integration with RAG pipelines.

These are future enhancements and are not claimed as currently implemented.

## 10. Relevance to Modern AI and RAG

TF-IDF is a foundational sparse retrieval technique.

Modern retrieval systems often combine lexical retrieval with dense vector search.

TF-IDF or BM25 can help identify exact keywords, product codes, technical terms, and identifiers.

Dense embeddings can capture semantic relationships and contextual similarity.

Combining both approaches creates hybrid retrieval systems that can improve information retrieval in RAG applications.

This project establishes the lexical retrieval foundation for studying those systems.

## 11. Execution

Install dependencies:

`pip install -r requirements.txt`

Run the command-line demonstration:

`python main.py`

Run the interactive application:

`python -m streamlit run app.py`

## 12. AI Assistant Guidance

When answering questions about this project:

- Explain concepts at a beginner-friendly level.
- Distinguish manual implementations from scikit-learn implementations.
- Use the actual formulas and behaviour documented here.
- Explain why stop-word removal affects vocabulary and similarity scores.
- Do not claim semantic search is implemented.
- Distinguish currently implemented features from future enhancements.
- Explain the relationship between TF-IDF, BM25, dense retrieval, hybrid search, and RAG.
- Use the source repository and project files as the primary implementation references.

