"""
Word2Vec Skip-Gram with Negative Sampling (SGNS) - pure NumPy.

Gradient derivations
====================
Let  u_c  = W_in[centre]       (centre word embedding, shape (d,))
     v_o  = W_out[context]     (positive context embedding, shape (d,))
     v_nk = W_out[neg_k]       (k-th negative sample embedding, shape (d,))
     σ    = sigmoid

The SGNS loss for one (centre, context) pair with K negatives is:

    L = -log σ(v_o · u_c) - Σ_k log σ(-v_nk · u_c)

Gradients (analytic):

  ∂L/∂u_c   = (σ(v_o · u_c) - 1) · v_o  +  Σ_k σ(v_nk · u_c) · v_nk
  ∂L/∂v_o   = (σ(v_o · u_c) - 1) · u_c
  ∂L/∂v_nk  = σ(v_nk · u_c) · u_c        for each k

Proof sketch for ∂L/∂u_c (positive term):
    ∂/∂u_c [-log σ(v_o · u_c)]
  = -1/σ(s_o) · σ(s_o)(1 - σ(s_o)) · v_o      (chain rule, s_o = v_o · u_c)
  = -(1 - σ(s_o)) · v_o
  = (σ(s_o) - 1) · v_o

Proof sketch for ∂L/∂u_c (negative term, k-th sample):
    ∂/∂u_c [-log σ(-v_nk · u_c)]
  = -1/σ(-s_k) · σ(-s_k)(1 - σ(-s_k)) · (-v_nk)   (chain rule, s_k = v_nk · u_c)
  = -(1 - σ(-s_k)) · (-v_nk)
  = σ(s_k) · v_nk                                    (since 1 - σ(-x) = σ(x))
"""

import numpy as np


def sigmoid(x):
    """Numerically stable sigmoid: clips input to [-30, 30] to avoid overflow."""
    x = np.clip(x, -30, 30)  # prevent exp overflow in float64
    return 1.0 / (1.0 + np.exp(-x))


class Word2Vec:
    """Skip-Gram with Negative Sampling, trained via vanilla SGD.

    Parameters
    ----------
    vocab_size : int
        Number of words in the vocabulary.
    embed_dim : int
        Dimensionality of word embeddings.
    """

    def __init__(self, vocab_size, embed_dim):
        # Uniform init in [-0.5/d, 0.5/d] - keeps initial dot products small
        bound = 0.5 / embed_dim
        self.W_in = np.random.uniform(-bound, bound,
                                      (vocab_size, embed_dim))   # centre word embeddings
        self.W_out = np.random.uniform(-bound, bound,
                                       (vocab_size, embed_dim))  # context / output embeddings

    def train_pair(self, centre_idx, context_idx, neg_indices, lr):
        """Train on a single (centre, context) pair with negative samples.

        Parameters
        ----------
        centre_idx : int
            Index of the centre word.
        context_idx : int
            Index of the true context word (positive sample).
        neg_indices : np.ndarray of int, shape (K,)
            Indices of K negative samples.
        lr : float
            Current learning rate.

        Returns
        -------
        float
            The SGNS loss for this pair.
        """
        # --- Forward pass ---
        u_c = self.W_in[centre_idx]          # centre embedding, shape (d,)
        v_o = self.W_out[context_idx]        # positive context embedding, shape (d,)
        V_n = self.W_out[neg_indices]        # negative embeddings, shape (K, d)

        s_o = np.dot(v_o, u_c)              # positive score (scalar)
        s_n = V_n @ u_c                     # negative scores, shape (K,)

        sig_o = sigmoid(s_o)                 # σ(s_o) - want this close to 1
        sig_n = sigmoid(s_n)                 # σ(s_k) - want these close to 0

        # --- Loss: -log σ(s_o) - Σ log σ(-s_k) ---
        loss = -np.log(sig_o + 1e-10) - np.sum(np.log(1.0 - sig_n + 1e-10))

        # --- Gradients (exact analytic, derived in module docstring) ---
        # ∂L/∂u_c = (σ(s_o) - 1)*v_o + Σ_k σ(s_k)*v_nk
        grad_u_c = (sig_o - 1.0) * v_o + (sig_n[:, None] * V_n).sum(axis=0)

        # ∂L/∂v_o = (σ(s_o) - 1)*u_c
        grad_v_o = (sig_o - 1.0) * u_c

        # ∂L/∂v_nk = σ(s_k)*u_c  for each k
        grad_V_n = sig_n[:, None] * u_c[None, :]  # shape (K, d)

        # --- SGD updates: θ ← θ - lr * ∂L/∂θ ---
        self.W_in[centre_idx] -= lr * grad_u_c       # update centre embedding
        self.W_out[context_idx] -= lr * grad_v_o      # update positive context embedding

        # np.add.at accumulates gradients for duplicate indices correctly,
        # unlike W_out[neg_indices] -= ... which would only apply the last
        # update when an index appears more than once in neg_indices.
        np.add.at(self.W_out, neg_indices, -lr * grad_V_n)

        return float(loss)

    def get_embeddings(self):
        """Return final embeddings as the average of W_in and W_out.

        Averaging both matrices tends to give slightly better results than
        using W_in alone, as noted by Mikolov et al.
        """
        return (self.W_in + self.W_out) / 2.0

    def most_similar(self, word, word2idx, idx2word, topn=10):
        """Find the topn most similar words to `word` by cosine similarity.

        Uses W_in (centre embeddings) for the lookup.

        Parameters
        ----------
        word : str
        word2idx : dict
        idx2word : dict
        topn : int

        Returns
        -------
        list of (str, float)
            Top-n most similar words with their cosine similarities.
        """
        if word not in word2idx:
            return []

        idx = word2idx[word]
        vec = self.W_in[idx]                              # query vector, shape (d,)

        # Cosine similarity against all words: cos(a,b) = (a·b) / (|a|·|b|)
        norms = np.linalg.norm(self.W_in, axis=1)         # shape (V,)
        norms = np.maximum(norms, 1e-10)                   # avoid division by zero
        similarities = self.W_in @ vec / (norms * np.linalg.norm(vec) + 1e-10)

        # Exclude the query word itself, then take top-n
        similarities[idx] = -np.inf
        top_indices = np.argsort(similarities)[::-1][:topn]

        return [(idx2word[i], float(similarities[i])) for i in top_indices]
