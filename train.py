"""
Training loop for Word2Vec SGNS on text8.

Hyperparameters follow Mikolov et al. 2013:
    embed_dim   = 100       embedding dimensionality
    window      = 5         max context window radius
    neg_samples = 5         negative samples per positive pair
    lr_init     = 0.025     initial learning rate
    lr_min      = 1e-4      minimum learning rate (floor for linear decay)
    epochs      = 1         number of passes over the corpus
"""

import numpy as np
from tqdm import tqdm

from data import load
from model import Word2Vec

# --- Hyperparameters ---
EMBED_DIM = 100
WINDOW = 5
NEG_SAMPLES = 5
LR_INIT = 0.025
LR_MIN = 1e-4
EPOCHS = 1


def iter_centre_context(token_ids, window):
    """Yield (centre_idx, context_idx) pairs with a dynamic window.

    For each centre word, the actual window radius is sampled uniformly
    from [1, window].  This effectively weights nearby context words
    more heavily than distant ones: a word at distance 1 is always
    included, while a word at distance `window` is included only 1/window
    of the time.

    Parameters
    ----------
    token_ids : np.ndarray of int
    window : int
        Maximum window radius.

    Yields
    ------
    (int, int)
        (centre_word_id, context_word_id)
    """
    n = len(token_ids)
    for i in range(n):
        # Dynamic window: sample radius from Uniform[1, window]
        radius = np.random.randint(1, window + 1)
        left = max(0, i - radius)
        right = min(n, i + radius + 1)
        centre = token_ids[i]
        for j in range(left, right):
            if j != i:
                yield centre, token_ids[j]


def train():
    """Main training function."""
    # --- Load and preprocess data ---
    token_ids, word2idx, idx2word, noise_dist = load()
    vocab_size = len(word2idx)

    # --- Initialise model ---
    model = Word2Vec(vocab_size, EMBED_DIM)

    # Estimate total pairs for LR scheduling and progress bar
    # Average window radius = (1+window)/2, each gives ~2*radius context words
    avg_radius = (1 + WINDOW) / 2.0
    est_pairs = int(len(token_ids) * 2 * avg_radius) * EPOCHS
    print(f"Estimated training pairs: {est_pairs:,}")

    step = 0
    running_loss = 0.0
    log_every = 100_000

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        pbar = tqdm(iter_centre_context(token_ids, WINDOW),
                    total=est_pairs // EPOCHS,
                    desc=f"Epoch {epoch+1}", unit="pair")

        for centre, context in pbar:
            # --- Linear LR decay from lr_init to lr_min ---
            progress = step / est_pairs
            lr = max(LR_MIN, LR_INIT * (1.0 - progress))

            # --- Sample negative indices from noise distribution ---
            neg_indices = np.random.choice(
                vocab_size, size=NEG_SAMPLES, replace=False, p=noise_dist
            )

            # --- Train one (centre, context) pair ---
            loss = model.train_pair(centre, context, neg_indices, lr)

            running_loss += loss
            step += 1

            if step % log_every == 0:
                avg_loss = running_loss / log_every
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.5f}")
                running_loss = 0.0

        pbar.close()

    # --- Save embeddings and vocab ---
    np.save("embeddings_in.npy", model.W_in)
    np.save("embeddings_out.npy", model.W_out)
    with open("vocab.txt", "w") as f:
        for i in range(vocab_size):
            f.write(idx2word[i] + "\n")
    print("\nSaved embeddings_in.npy, embeddings_out.npy, vocab.txt")

    # --- Quick evaluation ---
    print("\n--- Most similar words ---")
    for query in ["king", "paris"]:
        if query in word2idx:
            similar = model.most_similar(query, word2idx, idx2word, topn=10)
            print(f"\n  {query}:")
            for w, s in similar:
                print(f"    {w:15s} {s:.4f}")
        else:
            print(f"  '{query}' not in vocabulary")


if __name__ == "__main__":
    train()
