"""
Gradient checking and unit tests for word2vec SGNS.

Uses finite-difference approximation to verify that the analytic gradients
in model.py are correct.  This is the standard way to debug backprop:

    ∂L/∂θ_i  ≈  [L(θ_i + ε) - L(θ_i - ε)] / (2ε)

If the relative error between analytic and numerical gradients is < 1e-5,
the implementation is almost certainly correct.

Run:  python tests.py
"""

import numpy as np
from model import Word2Vec, sigmoid


# ---------------------------------------------------------------------------
# Helper: compute SGNS loss without mutating any weights
# ---------------------------------------------------------------------------

def compute_loss(W_in, W_out, centre_idx, context_idx, neg_indices):
    """Pure function - computes SGNS loss from weight matrices."""
    u_c = W_in[centre_idx]
    v_o = W_out[context_idx]
    V_n = W_out[neg_indices]

    s_o = np.dot(v_o, u_c)
    s_n = V_n @ u_c

    sig_o = sigmoid(s_o)
    sig_n = sigmoid(s_n)

    loss = -np.log(sig_o + 1e-10) - np.sum(np.log(1.0 - sig_n + 1e-10))
    return float(loss)


# ---------------------------------------------------------------------------
# Helper: compute analytic gradients without applying SGD updates
# ---------------------------------------------------------------------------

def compute_analytic_gradients(W_in, W_out, centre_idx, context_idx, neg_indices):
    """Return (grad_u_c, grad_v_o, grad_V_n) without modifying weights."""
    u_c = W_in[centre_idx]
    v_o = W_out[context_idx]
    V_n = W_out[neg_indices]

    s_o = np.dot(v_o, u_c)
    s_n = V_n @ u_c

    sig_o = sigmoid(s_o)
    sig_n = sigmoid(s_n)

    grad_u_c = (sig_o - 1.0) * v_o + (sig_n[:, None] * V_n).sum(axis=0)
    grad_v_o = (sig_o - 1.0) * u_c
    grad_V_n = sig_n[:, None] * u_c[None, :]

    return grad_u_c, grad_v_o, grad_V_n


# ---------------------------------------------------------------------------
# Finite-difference gradient check
# ---------------------------------------------------------------------------

def numerical_gradient(W_in, W_out, centre_idx, context_idx, neg_indices,
                       matrix, row, eps=1e-5):
    """Compute numerical gradient for matrix[row] via central differences.

    Parameters
    ----------
    matrix : np.ndarray
        Reference to either W_in or W_out (will be mutated and restored).
    row : int
        Which row (word index) to differentiate w.r.t.
    eps : float
        Perturbation size.

    Returns
    -------
    np.ndarray, shape (embed_dim,)
    """
    d = matrix.shape[1]
    grad = np.zeros(d)
    for j in range(d):
        orig = matrix[row, j]

        matrix[row, j] = orig + eps
        loss_plus = compute_loss(W_in, W_out, centre_idx, context_idx, neg_indices)

        matrix[row, j] = orig - eps
        loss_minus = compute_loss(W_in, W_out, centre_idx, context_idx, neg_indices)

        grad[j] = (loss_plus - loss_minus) / (2 * eps)
        matrix[row, j] = orig  # restore

    return grad


def relative_error(a, b):
    """Max relative error between two vectors, safe for near-zero values."""
    return np.max(np.abs(a - b) / (np.maximum(np.abs(a) + np.abs(b), 1e-8)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_gradient_check():
    """Verify analytic gradients match finite-difference approximation."""
    np.random.seed(42)
    vocab_size, embed_dim = 20, 8
    model = Word2Vec(vocab_size, embed_dim)

    centre_idx = 0
    context_idx = 3
    neg_indices = np.array([5, 7, 12])

    # Analytic gradients
    grad_u_c, grad_v_o, grad_V_n = compute_analytic_gradients(
        model.W_in, model.W_out, centre_idx, context_idx, neg_indices
    )

    # Numerical gradient for u_c (W_in[centre_idx])
    num_grad_u_c = numerical_gradient(
        model.W_in, model.W_out, centre_idx, context_idx, neg_indices,
        matrix=model.W_in, row=centre_idx
    )
    err = relative_error(grad_u_c, num_grad_u_c)
    print(f"  grad_u_c   relative error: {err:.2e}  {'PASS' if err < 1e-5 else 'FAIL'}")
    assert err < 1e-5, f"grad_u_c check failed: rel error {err:.2e}"

    # Numerical gradient for v_o (W_out[context_idx])
    num_grad_v_o = numerical_gradient(
        model.W_in, model.W_out, centre_idx, context_idx, neg_indices,
        matrix=model.W_out, row=context_idx
    )
    err = relative_error(grad_v_o, num_grad_v_o)
    print(f"  grad_v_o   relative error: {err:.2e}  {'PASS' if err < 1e-5 else 'FAIL'}")
    assert err < 1e-5, f"grad_v_o check failed: rel error {err:.2e}"

    # Numerical gradient for each negative sample v_nk (W_out[neg_indices[k]])
    for k, neg_idx in enumerate(neg_indices):
        num_grad_v_nk = numerical_gradient(
            model.W_in, model.W_out, centre_idx, context_idx, neg_indices,
            matrix=model.W_out, row=neg_idx
        )
        err = relative_error(grad_V_n[k], num_grad_v_nk)
        print(f"  grad_V_n[{k}] relative error: {err:.2e}  {'PASS' if err < 1e-5 else 'FAIL'}")
        assert err < 1e-5, f"grad_V_n[{k}] check failed: rel error {err:.2e}"


def test_loss_decreases():
    """Verify that repeated training on the same pair decreases loss."""
    np.random.seed(123)
    model = Word2Vec(10, 4)
    centre, context, negs = 0, 1, np.array([2, 3, 4])

    losses = [model.train_pair(centre, context, negs, lr=0.1) for _ in range(50)]

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
    print(f"  loss: {losses[0]:.4f} -> {losses[-1]:.4f}  PASS")


def test_sgd_updates_correct_rows():
    """Verify that train_pair only modifies the relevant embedding rows."""
    np.random.seed(7)
    model = Word2Vec(10, 4)
    W_in_before = model.W_in.copy()
    W_out_before = model.W_out.copy()

    centre, context, negs = 2, 5, np.array([0, 8])
    model.train_pair(centre, context, negs, lr=0.01)

    # W_in: only row `centre` should change
    changed_in = set(np.where(np.any(model.W_in != W_in_before, axis=1))[0])
    assert changed_in == {centre}, f"W_in changed rows {changed_in}, expected {{{centre}}}"
    print(f"  W_in  changed rows: {changed_in}  PASS")

    # W_out: only rows `context` and `negs` should change
    changed_out = set(np.where(np.any(model.W_out != W_out_before, axis=1))[0])
    expected_out = {context} | set(negs)
    assert changed_out == expected_out, (
        f"W_out changed rows {changed_out}, expected {expected_out}"
    )
    print(f"  W_out changed rows: {changed_out}  PASS")


def test_build_vocab():
    """Verify vocabulary construction with min_count filtering."""
    from data import build_vocab

    tokens = "the cat sat on the mat the cat the dog".split()
    w2i, i2w, counts = build_vocab(tokens, min_count=2)

    # "the" appears 4x, "cat" 2x - both should be in vocab
    # "sat", "on", "mat", "dog" appear 1x - should be filtered out
    assert "the" in w2i, "'the' missing from vocab"
    assert "cat" in w2i, "'cat' missing from vocab"
    assert "sat" not in w2i, "'sat' should be filtered (count=1)"
    assert "dog" not in w2i, "'dog' should be filtered (count=1)"

    # Index 0 should be the most frequent word
    assert i2w[0] == "the", f"Expected 'the' at index 0, got '{i2w[0]}'"
    assert counts["the"] == 4
    assert counts["cat"] == 2
    print(f"  vocab: {w2i}  PASS")


def test_noise_distribution():
    """Verify noise distribution sums to 1 and respects the 0.75 power."""
    from data import build_vocab, make_noise_distribution

    tokens = "a a a a b b c".split()
    w2i, i2w, counts = build_vocab(tokens, min_count=1)
    dist = make_noise_distribution(counts)

    assert abs(dist.sum() - 1.0) < 1e-10, f"Distribution sums to {dist.sum()}"
    assert len(dist) == len(w2i)
    # Most frequent word ("a") should have highest probability
    assert dist[w2i["a"]] > dist[w2i["b"]] > dist[w2i["c"]]
    print(f"  noise dist: {dist}  sums to {dist.sum():.10f}  PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("GRADIENT CHECK (finite differences)")
    print("=" * 50)
    test_gradient_check()

    print("\n" + "=" * 50)
    print("LOSS CONVERGENCE")
    print("=" * 50)
    test_loss_decreases()

    print("\n" + "=" * 50)
    print("SGD UPDATE CORRECTNESS")
    print("=" * 50)
    test_sgd_updates_correct_rows()

    print("\n" + "=" * 50)
    print("VOCABULARY CONSTRUCTION")
    print("=" * 50)
    test_build_vocab()

    print("\n" + "=" * 50)
    print("NOISE DISTRIBUTION")
    print("=" * 50)
    test_noise_distribution()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
