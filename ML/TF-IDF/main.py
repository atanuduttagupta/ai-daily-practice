
"""Manual Bag-of-Words and TF-IDF implementation.

Supports configurable stop-word removal, cosine similarity search,
and validation against scikit-learn.
"""

import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import (
    CountVectorizer,
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)


TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def tokenize(text: str, stop_words=None) -> list[str]:
    """Convert text into lowercase tokens and optionally remove stop words."""

    tokens = TOKEN_RE.findall(text.lower())

    if stop_words is not None:
        stop_words = set(stop_words)
        tokens = [token for token in tokens if token not in stop_words]

    return tokens


def build_vocab(docs: list[str], stop_words=None) -> list[str]:
    """Build a sorted vocabulary from the document corpus."""

    return sorted({
        token
        for document in docs
        for token in tokenize(document, stop_words)
    })


def bag_of_words(
    docs: list[str],
    vocab: list[str],
    stop_words=None,
) -> np.ndarray:
    """Create a document-term matrix using raw token counts."""

    index = {word: i for i, word in enumerate(vocab)}

    matrix = np.zeros((len(docs), len(vocab)))

    for row, document in enumerate(docs):
        counts = Counter(tokenize(document, stop_words))

        for token, count in counts.items():
            if token in index:
                matrix[row, index[token]] = count

    return matrix


def calculate_idf(counts: np.ndarray) -> np.ndarray:
    """Calculate scikit-learn-compatible smoothed IDF."""

    n_docs = counts.shape[0]

    if n_docs == 0:
        return np.ones(counts.shape[1])

    document_frequency = (counts > 0).sum(axis=0)

    return np.log(
        (1 + n_docs) / (1 + document_frequency)
    ) + 1


def tfidf(counts: np.ndarray, idf=None) -> np.ndarray:
    """Calculate TF-IDF and apply L2 normalization.

    Uses raw term counts multiplied by smoothed IDF.
    """

    if idf is None:
        idf = calculate_idf(counts)

    weighted = counts * idf

    norms = np.linalg.norm(weighted, axis=1, keepdims=True)

    return np.divide(
        weighted,
        norms,
        out=np.zeros_like(weighted, dtype=float),
        where=norms != 0,
    )


def cosine_scores(
    query_counts: np.ndarray,
    document_matrix: np.ndarray,
    idf: np.ndarray,
) -> np.ndarray:
    """Calculate cosine similarity using corpus IDF values."""

    query_vector = query_counts * idf

    norm = np.linalg.norm(query_vector)

    if norm == 0:
        return np.zeros(document_matrix.shape[0])

    query_vector = query_vector / norm

    return document_matrix @ query_vector


def search_documents(
    docs: list[str],
    query: str,
    remove_stop_words: bool = True,
):
    """Rank documents according to manual TF-IDF cosine similarity."""

    if not docs:
        raise ValueError("Document corpus cannot be empty.")

    stop_words = ENGLISH_STOP_WORDS if remove_stop_words else None

    vocab = build_vocab(docs, stop_words)

    if not vocab:
        raise ValueError(
            "No usable vocabulary. Check the documents and stop-word setting."
        )

    counts = bag_of_words(docs, vocab, stop_words)

    idf = calculate_idf(counts)

    document_weights = tfidf(counts, idf)

    query_counts = bag_of_words([query], vocab, stop_words)[0]

    scores = cosine_scores(
        query_counts,
        document_weights,
        idf,
    )

    ranked_indices = np.argsort(scores)[::-1]

    return {
        "vocabulary": vocab,
        "counts": counts,
        "idf": idf,
        "tfidf": document_weights,
        "scores": scores,
        "ranked_indices": ranked_indices,
    }


def validate_against_sklearn(
    docs: list[str],
    remove_stop_words: bool = True,
) -> bool:
    """Verify manual BoW and TF-IDF against scikit-learn."""

    stop_words = ENGLISH_STOP_WORDS if remove_stop_words else None
    sklearn_stop_words = "english" if remove_stop_words else None

    vocab = build_vocab(docs, stop_words)

    if not vocab:
        raise ValueError("No usable vocabulary for validation.")

    counts = bag_of_words(docs, vocab, stop_words)

    manual_weights = tfidf(counts)

    count_vectorizer = CountVectorizer(
        stop_words=sklearn_stop_words,
    )

    sklearn_counts = count_vectorizer.fit_transform(docs).toarray()

    assert list(count_vectorizer.get_feature_names_out()) == vocab
    assert np.array_equal(sklearn_counts, counts)

    tfidf_vectorizer = TfidfVectorizer(
        stop_words=sklearn_stop_words,
    )

    sklearn_weights = tfidf_vectorizer.fit_transform(docs).toarray()

    assert list(tfidf_vectorizer.get_feature_names_out()) == vocab
    assert np.allclose(sklearn_weights, manual_weights)

    return True


def main():
    """Run the original command-line demonstration."""

    docs = [
        "the cat sat on the mat",
        "the dog chased the cat",
        "dogs and cats are popular pets",
        "stock markets fell sharply on monday",
        "investors watch the stock market closely",
    ]

    query = "stock market news"

    print("TF-IDF Document Search")
    print("=" * 40)

    for remove_stop_words in (False, True):

        print(
            f"\nStop-word removal: "
            f"{'ON' if remove_stop_words else 'OFF'}"
        )

        result = search_documents(
            docs,
            query,
            remove_stop_words=remove_stop_words,
        )

        print(f"Vocabulary size: {len(result['vocabulary'])}")

        print(
            f"Bag-of-words matrix shape: "
            f"{result['counts'].shape}"
        )

        validate_against_sklearn(
            docs,
            remove_stop_words=remove_stop_words,
        )

        print("Manual implementation matches scikit-learn.")

        print(f'\nQuery: "{query}"')

        for rank, index in enumerate(
            result["ranked_indices"],
            start=1,
        ):
            print(
                f"{rank}. "
                f"score={result['scores'][index]:.3f} "
                f"{docs[index]}"
            )


if __name__ == "__main__":
    main()
