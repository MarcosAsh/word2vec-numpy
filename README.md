# word2vec from scratch — pure NumPy

A complete implementation of **Word2Vec Skip-Gram with Negative Sampling (SGNS)** using only NumPy. No PyTorch, TensorFlow, or any ML framework — just matrix operations and SGD.

Trained on the [text8](http://mattmahoney.net/dc/text8.zip) corpus (~17M tokens of cleaned Wikipedia text).

## Files

| File | Description |
|------|-------------|
| `model.py` | Core `Word2Vec` class: embeddings, forward pass, analytic gradients, SGD updates |
| `data.py` | Downloads text8, builds vocabulary, subsamples frequent words, creates noise distribution |
| `train.py` | Training loop with linear LR decay, dynamic windowing, negative sampling |
| `eval.py` | Nearest-neighbour search and word analogy evaluation (3CosAdd) |

## Quickstart

```bash
pip install numpy tqdm
python train.py   # downloads text8, trains 1 epoch (~30-60 min on CPU)
python eval.py    # nearest neighbours + analogy evaluation
```

## Theory

### Skip-gram objective

The skip-gram model aims to learn word embeddings by predicting context words given a centre word. The softmax objective for a centre word $w_c$ and context word $w_o$ over vocabulary $V$ is:

$$P(w_o \mid w_c) = \frac{\exp(v_o^\top u_c)}{\sum_{w \in V} \exp(v_w^\top u_c)}$$

Computing the full softmax is prohibitively expensive ($O(|V|)$ per pair), so we use **negative sampling** instead.

### Negative sampling loss

For a positive (centre, context) pair and $K$ negative samples drawn from a noise distribution $P_n$, the SGNS loss is:

$$L = -\log \sigma(v_o^\top u_c) - \sum_{k=1}^{K} \log \sigma(-v_{n_k}^\top u_c)$$

where $\sigma(x) = 1/(1 + e^{-x})$ is the sigmoid function. This objective pushes the model to:
- Score the true context word **high** (first term)
- Score random noise words **low** (second term)

### Gradient derivations

Starting from the loss $L = -\log \sigma(s_o) - \sum_k \log \sigma(-s_k)$ where $s_o = v_o^\top u_c$ and $s_k = v_{n_k}^\top u_c$:

**Gradient w.r.t. centre embedding $u_c$:**

$$\frac{\partial L}{\partial u_c} = (\sigma(s_o) - 1) \cdot v_o + \sum_{k=1}^{K} \sigma(s_k) \cdot v_{n_k}$$

*Derivation (positive term):*

$$\frac{\partial}{\partial u_c}[-\log \sigma(s_o)] = -\frac{\sigma(s_o)(1-\sigma(s_o))}{\sigma(s_o)} \cdot v_o = -(1 - \sigma(s_o)) \cdot v_o = (\sigma(s_o) - 1) \cdot v_o$$

*Derivation (negative term, for each $k$):*

$$\frac{\partial}{\partial u_c}[-\log \sigma(-s_k)] = -\frac{\sigma(-s_k)(1-\sigma(-s_k))}{\sigma(-s_k)} \cdot (-v_{n_k}) = (1 - \sigma(-s_k)) \cdot v_{n_k} = \sigma(s_k) \cdot v_{n_k}$$

using the identity $1 - \sigma(-x) = \sigma(x)$.

**Gradient w.r.t. positive context embedding $v_o$:**

$$\frac{\partial L}{\partial v_o} = (\sigma(s_o) - 1) \cdot u_c$$

**Gradient w.r.t. negative sample embedding $v_{n_k}$:**

$$\frac{\partial L}{\partial v_{n_k}} = \sigma(s_k) \cdot u_c$$

## Design decisions

### Two embedding matrices ($W_{in}$ vs $W_{out}$)

We maintain separate embedding matrices for centre words ($W_{in}$) and context words ($W_{out}$). This avoids the degenerate solution where the model simply makes every word similar to itself. Averaging both matrices at the end (`get_embeddings()`) yields slightly better representations.

### Dynamic window

Instead of a fixed context window, we sample the actual radius uniformly from $[1, \text{window}]$ for each centre word. This implicitly weights nearby words more: a word at distance 1 is **always** a context word, while a word at distance 5 is included only $1/5$ of the time. This mirrors the intuition that immediately adjacent words carry stronger co-occurrence signal.

### Sub-sampling of frequent words

Very frequent words like "the" and "a" co-occur with nearly everything but provide little semantic information. We discard each token with probability:

$$P_{\text{discard}}(w) = 1 - \sqrt{\frac{t}{f(w)}}$$

where $f(w)$ is the word's relative frequency and $t$ is a threshold (default $10^{-5}$). This aggressively removes the most common words while keeping rare words nearly intact.

### Noise distribution: why $\text{count}^{0.75}$

The noise distribution for negative sampling raises unigram counts to the power $0.75$:

$$P_n(w) \propto \text{count}(w)^{0.75}$$

This smooths the distribution compared to raw frequencies: rare words get sampled as negatives more often (providing more training signal for them), while extremely common words are down-weighted. The exponent $0.75$ was found empirically by Mikolov et al. to outperform both $1.0$ (raw frequencies) and uniform sampling.

### Linear learning rate decay

The learning rate decays linearly from `lr_init` (0.025) to `lr_min` (0.0001) over the course of training. This aggressive schedule allows large updates early (fast convergence to a good region) while preventing oscillation near the end. A hard floor at `lr_min` ensures the model can still learn from late examples.

### `np.add.at` for negative gradients

When updating negative sample embeddings, duplicate indices can occur (the same word sampled as a negative more than once). A naive `W_out[neg_indices] -= lr * grad` would only apply the **last** update for duplicated indices. `np.add.at` correctly **accumulates** all gradient contributions, ensuring mathematically correct SGD updates.

## Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| `embed_dim` | 100 | Mikolov et al. 2013 |
| `window` | 5 | Mikolov et al. 2013 |
| `neg_samples` | 5 | Mikolov et al. 2013 |
| `lr_init` | 0.025 | Mikolov et al. 2013 |
| `sub-sampling t` | $10^{-5}$ | Mikolov et al. 2013 |
| `noise power` | 0.75 | Mikolov et al. 2013 |
| `min_count` | 5 | Mikolov et al. 2013 / word2vec.c default |

## References

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781). *arXiv:1301.3781*.
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546). *NeurIPS 2013*.
3. Original C implementation: [word2vec.c](https://github.com/tmikolov/word2vec/blob/master/word2vec.c)
