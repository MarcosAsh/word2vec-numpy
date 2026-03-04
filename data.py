"""
Data loading and preprocessing for the text8 corpus.

text8 is a cleaned dump of English Wikipedia (~100 MB of lowercase text,
no punctuation), commonly used as a benchmark for word embedding models.
"""

import os
import zipfile
import urllib.request
from collections import Counter

import numpy as np


TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
TEXT8_ZIP = "text8.zip"
TEXT8_FILE = "text8"


def download_text8():
    """Download and extract text8 corpus if not already present."""
    if not os.path.exists(TEXT8_FILE):
        if not os.path.exists(TEXT8_ZIP):
            print(f"Downloading {TEXT8_URL} ...")
            urllib.request.urlretrieve(TEXT8_URL, TEXT8_ZIP)
            print("Download complete.")
        with zipfile.ZipFile(TEXT8_ZIP, "r") as zf:
            zf.extractall(".")
        print("Extracted text8.")
    return open(TEXT8_FILE).read().split()


def build_vocab(tokens, min_count=5):
    """Build vocabulary from token list.

    Parameters
    ----------
    tokens : list of str
        Raw token sequence.
    min_count : int
        Discard words appearing fewer than min_count times.

    Returns
    -------
    word2idx : dict
        Maps word → integer index (sorted by descending frequency).
    idx2word : dict
        Maps integer index → word.
    counts : dict
        Maps word → raw count (only for words in vocab).
    """
    raw_counts = Counter(tokens)
    # Sort by frequency descending - most common word gets index 0
    sorted_words = sorted(
        [w for w, c in raw_counts.items() if c >= min_count],
        key=lambda w: raw_counts[w],
        reverse=True,
    )
    word2idx = {w: i for i, w in enumerate(sorted_words)}
    idx2word = {i: w for w, i in word2idx.items()}
    counts = {w: raw_counts[w] for w in sorted_words}
    return word2idx, idx2word, counts


def subsample(tokens, word2idx, counts, t=1e-5):
    """Subsample frequent words following Mikolov et al. 2013.

    Each token is kept with probability  1 - sqrt(t / f(w))
    where f(w) = count(w) / total_tokens.  This down-weights very
    frequent words (like "the", "a") that carry little semantic signal.

    Parameters
    ----------
    tokens : list of str
    word2idx : dict
    counts : dict
    t : float
        Threshold controlling aggressiveness of subsampling.
        Smaller t → more aggressive subsampling.

    Returns
    -------
    list of str
        Filtered token list.
    """
    total = sum(counts.values())
    keep = []
    for w in tokens:
        if w not in word2idx:
            continue
        freq = counts[w] / total
        # Keep with probability sqrt(t / f(w)); discard with 1 - sqrt(t / f(w))
        p_keep = np.sqrt(t / freq) if freq > t else 1.0
        if np.random.random() < p_keep:
            keep.append(w)
    return keep


def make_noise_distribution(counts, power=0.75):
    """Create the unigram noise distribution raised to the 0.75 power.

    Raising counts to 0.75 (instead of using raw frequencies) smooths
    the distribution: rare words get sampled more often than their
    frequency would suggest, and common words less often.  This was
    found empirically by Mikolov et al. to give the best results.

    Parameters
    ----------
    counts : dict
        Maps word → count (must be in index order).
    power : float
        Exponent applied to counts (default 0.75).

    Returns
    -------
    np.ndarray, shape (vocab_size,)
        Normalised noise distribution.
    """
    vocab_size = len(counts)
    dist = np.zeros(vocab_size, dtype=np.float64)
    # Build array aligned with word2idx (sorted by descending frequency)
    sorted_words = sorted(counts.keys(), key=lambda w: counts[w], reverse=True)
    for i, w in enumerate(sorted_words):
        dist[i] = counts[w] ** power
    dist /= dist.sum()
    return dist


def load(max_tokens=None, min_count=5, subsample_t=1e-5):
    """Load text8, build vocab, subsample, and return everything needed.

    Parameters
    ----------
    max_tokens : int or None
        If set, truncate the raw corpus to this many tokens.
    min_count : int
        Minimum word frequency to include in vocab.
    subsample_t : float
        Subsampling threshold.

    Returns
    -------
    token_ids : np.ndarray of int
        Subsampled token sequence as integer indices.
    word2idx : dict
    idx2word : dict
    noise_dist : np.ndarray
    """
    tokens = download_text8()
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    print(f"Raw tokens: {len(tokens):,}")

    word2idx, idx2word, counts = build_vocab(tokens, min_count=min_count)
    print(f"Vocab size (min_count={min_count}): {len(word2idx):,}")

    tokens = subsample(tokens, word2idx, counts, t=subsample_t)
    print(f"Tokens after subsampling: {len(tokens):,}")

    token_ids = np.array([word2idx[w] for w in tokens], dtype=np.int32)
    noise_dist = make_noise_distribution(counts)

    return token_ids, word2idx, idx2word, noise_dist
