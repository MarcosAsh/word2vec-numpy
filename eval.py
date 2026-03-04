"""
Evaluation utilities for word2vec embeddings.

Supports:
  - Nearest-neighbour lookup (cosine similarity)
  - Word analogy via the 3CosAdd method (Mikolov et al. 2013)
"""

import numpy as np
from model import Word2Vec


def analogy(a, b, c, word2idx, idx2word, embeddings, topn=5):
    """Solve the analogy  a : b :: c : ?  using the 3CosAdd method.

    Finds  d* = argmax_d  cos(d, b - a + c)
    excluding a, b, c from the candidate set.

    Parameters
    ----------
    a, b, c : str
        Words forming the analogy query (e.g. "king", "man", "woman").
    word2idx : dict
    idx2word : dict
    embeddings : np.ndarray, shape (V, d)
    topn : int

    Returns
    -------
    list of (str, float)
        Top-n results with cosine similarity scores.
    """
    for w in [a, b, c]:
        if w not in word2idx:
            print(f"  '{w}' not in vocabulary")
            return []

    # Target vector: b - a + c
    vec = embeddings[word2idx[b]] - embeddings[word2idx[a]] + embeddings[word2idx[c]]

    # Cosine similarity against all words
    norms = np.linalg.norm(embeddings, axis=1)
    norms = np.maximum(norms, 1e-10)
    similarities = embeddings @ vec / (norms * np.linalg.norm(vec) + 1e-10)

    # Exclude the query words
    for w in [a, b, c]:
        similarities[word2idx[w]] = -np.inf

    top_indices = np.argsort(similarities)[::-1][:topn]
    return [(idx2word[i], float(similarities[i])) for i in top_indices]


def run_eval(model, word2idx, idx2word):
    """Run standard evaluation: nearest neighbours + analogies.

    Parameters
    ----------
    model : Word2Vec or None
        If provided, uses model.W_in for nearest neighbours.
    word2idx : dict
    idx2word : dict
    """
    embeddings = model.get_embeddings()

    # --- Nearest neighbours ---
    print("=" * 50)
    print("NEAREST NEIGHBOURS (cosine similarity on W_in)")
    print("=" * 50)
    query_words = ["king", "france", "computer", "run", "good"]
    for word in query_words:
        similar = model.most_similar(word, word2idx, idx2word, topn=10)
        if similar:
            print(f"\n  {word}:")
            for w, s in similar:
                print(f"    {w:15s} {s:.4f}")
        else:
            print(f"\n  '{word}' not in vocabulary")

    # --- Analogies (3CosAdd) ---
    print("\n" + "=" * 50)
    print("ANALOGIES (3CosAdd)")
    print("=" * 50)
    analogy_tests = [
        ("king", "man", "woman", "queen"),
        ("paris", "france", "berlin", "germany"),
        ("good", "better", "bad", "worse"),
    ]
    for a, b, c, expected in analogy_tests:
        results = analogy(a, b, c, word2idx, idx2word, embeddings, topn=5)
        print(f"\n  {a} - {b} + {c} = ?  (expected: {expected})")
        for w, s in results:
            marker = " <--" if w == expected else ""
            print(f"    {w:15s} {s:.4f}{marker}")


if __name__ == "__main__":
    # Load saved embeddings and vocab
    print("Loading saved embeddings and vocabulary...")
    W_in = np.load("embeddings_in.npy")
    W_out = np.load("embeddings_out.npy")
    vocab_size, embed_dim = W_in.shape

    vocab = []
    with open("vocab.txt", "r") as f:
        for line in f:
            vocab.append(line.strip())

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}

    # Reconstruct model with loaded weights
    model = Word2Vec(vocab_size, embed_dim)
    model.W_in = W_in
    model.W_out = W_out

    run_eval(model, word2idx, idx2word)
