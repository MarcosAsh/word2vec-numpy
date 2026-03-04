# word2vec from scratch, pure NumPy

Word2Vec (Skip-Gram + Negative Sampling) in NumPy only. No PyTorch, TensorFlow, or any ML framework.

Trained on the [text8](http://mattmahoney.net/dc/text8.zip) corpus (~17M tokens of cleaned Wikipedia).

## Files

| File | Description |
|------|-------------|
| `model.py` | Word2Vec class: forward pass, analytic gradients, SGD |
| `data.py` | text8 download, vocab, subsampling, noise distribution |
| `train.py` | Training loop with LR decay, dynamic window, negative sampling |
| `eval.py` | Nearest neighbours + word analogy evaluation (3CosAdd) |
| `tests.py` | Finite-difference gradient check + unit tests |

## Quickstart

```bash
pip install -r requirements.txt
python tests.py   # gradient check + unit tests (~instant)
python train.py   # downloads text8, trains 1 epoch (~30-60 min on CPU)
python eval.py    # nearest neighbours + analogy evaluation
```

## Theory

### Skip-gram objective

Skip-gram learns embeddings by predicting context words from a centre word. The full softmax objective is:

$$P(w_o \mid w_c) = \frac{\exp(v_o^\top u_c)}{\sum_{w \in V} \exp(v_w^\top u_c)}$$

This is $O(|V|)$ per pair, so we use negative sampling instead.

### Negative sampling loss

For a positive pair and $K$ noise samples:

$$L = -\log \sigma(v_o^\top u_c) - \sum_{k=1}^{K} \log \sigma(-v_{n_k}^\top u_c)$$

First term pushes the true context score high, second term pushes noise scores low.

### Gradients

Let $s_o = v_o^\top u_c$, $s_k = v_{n_k}^\top u_c$.

$$\frac{\partial L}{\partial u_c} = (\sigma(s_o) - 1) \cdot v_o + \sum_{k=1}^{K} \sigma(s_k) \cdot v_{n_k}$$

$$\frac{\partial L}{\partial v_o} = (\sigma(s_o) - 1) \cdot u_c$$

$$\frac{\partial L}{\partial v_{n_k}} = \sigma(s_k) \cdot u_c$$

Derivation for the negative term uses $1 - \sigma(-x) = \sigma(x)$:

$$\frac{\partial}{\partial u_c}[-\log \sigma(-s_k)] = -(1 - \sigma(-s_k)) \cdot (-v_{n_k}) = \sigma(s_k) \cdot v_{n_k}$$

## Design decisions

**Two embedding matrices.** Separate $W_{in}$ (centre) and $W_{out}$ (context) avoids the degenerate solution where every word is most similar to itself. Final embeddings are the average of both.

**Dynamic window.** Radius sampled from $[1, \text{window}]$ per centre word. Words at distance 1 always appear as context; words at distance 5 appear only 1/5 of the time. This naturally up-weights nearby words.

**Sub-sampling.** Frequent words ("the", "a") are discarded with probability $1 - \sqrt{t/f(w)}$ where $t = 10^{-5}$. They co-occur with everything but carry little signal.

**Noise distribution.** $P_n(w) \propto \text{count}(w)^{0.75}$. The 0.75 exponent smooths the distribution so rare words get more negative-sampling exposure. Found empirically by Mikolov et al.

**Linear LR decay.** 0.025 down to 0.0001 over training. Big steps early, small steps late, with a floor so late examples still matter.

**`np.add.at` for negatives.** With duplicate negative indices, `W[idx] -= grad` only applies the last write. `np.add.at` accumulates correctly.

## Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| `embed_dim` | 100 | Mikolov et al. 2013 |
| `window` | 5 | Mikolov et al. 2013 |
| `neg_samples` | 5 | Mikolov et al. 2013 |
| `lr_init` | 0.025 | Mikolov et al. 2013 |
| `sub-sampling t` | $10^{-5}$ | Mikolov et al. 2013 |
| `noise power` | 0.75 | Mikolov et al. 2013 |
| `min_count` | 5 | Mikolov et al. 2013 / word2vec.c |

## References

1. Mikolov et al. (2013). [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781).
2. Mikolov et al. (2013). [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546).
3. [word2vec.c](https://github.com/tmikolov/word2vec/blob/master/word2vec.c) (original C implementation)
