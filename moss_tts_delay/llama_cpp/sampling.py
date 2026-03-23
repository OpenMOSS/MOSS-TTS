"""
Sampling functions for MOSS-TTS-Delay inference.

Pure NumPy ports of the PyTorch sampling utilities from
``moss_tts_delay/inference_utils.py``.
"""

from __future__ import annotations

import numpy as np


def apply_top_k(logits: np.ndarray, top_k: int) -> np.ndarray:
    """Keep only the top-k highest logits; set the rest to -inf."""
    N, V = logits.shape
    top_k = min(top_k, V)
    indices = np.argpartition(logits, -top_k, axis=-1)[:, -top_k:]
    filtered = np.full_like(logits, -np.inf)
    rows = np.arange(N)[:, None]
    filtered[rows, indices] = logits[rows, indices]
    return filtered


def apply_top_p(logits: np.ndarray, top_p: float) -> np.ndarray:
    """Nucleus sampling: keep tokens whose cumulative probability <= top_p."""
    max_logits = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_logits
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    sorted_idx = np.argsort(-probs, axis=-1)
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=-1)
    cum_probs = np.cumsum(sorted_probs, axis=-1)

    to_remove = cum_probs > top_p
    to_remove[:, 1:] = to_remove[:, :-1].copy()
    to_remove[:, 0] = False

    remove_original = np.zeros_like(logits, dtype=bool)
    np.put_along_axis(remove_original, sorted_idx, to_remove, axis=-1)

    result = logits.copy()
    result[remove_original] = -np.inf
    return result


def apply_repetition_penalty(
    logits: np.ndarray,
    prev_tokens: np.ndarray,
    penalty: float,
) -> np.ndarray:
    """Apply repetition penalty independently per VQ head."""
    if penalty == 1.0 or prev_tokens is None:
        return logits

    result = logits.copy()

    if result.ndim == 2:
        unique_tokens = np.unique(prev_tokens.ravel())
        unique_tokens = unique_tokens[unique_tokens >= 0]
        if len(unique_tokens) == 0:
            return result
        token_logits = result[:, unique_tokens]
        pos_mask = token_logits > 0
        token_logits[pos_mask] /= penalty
        token_logits[~pos_mask] *= penalty
        result[:, unique_tokens] = token_logits
        return result

    assert result.ndim == 3, "Audio logits must be (B, H, V)"
    _B, H, _V = result.shape
    for h in range(H):
        prev_h = prev_tokens[..., h].ravel()
        unique_tokens = np.unique(prev_h)
        unique_tokens = unique_tokens[unique_tokens >= 0]
        if len(unique_tokens) == 0:
            continue
        token_logits = result[:, h, unique_tokens]
        pos_mask = token_logits > 0
        token_logits[pos_mask] /= penalty
        token_logits[~pos_mask] *= penalty
        result[:, h, unique_tokens] = token_logits

    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def multinomial(probs: np.ndarray) -> np.ndarray:
    """Sample one token per row via multinomial distribution."""
    N = probs.shape[0]
    cum = np.cumsum(probs, axis=-1)
    r = np.random.random(N)[:, None]
    # Clamp to valid range: float32 cumsum may not reach 1.0 exactly,
    # so r > cumsum[-1] would produce an out-of-bounds index.
    return np.minimum((cum < r).sum(axis=-1), probs.shape[-1] - 1).astype(np.int64)


def sample_token(
    logits: np.ndarray,
    prev_tokens: np.ndarray | None = None,
    repetition_penalty: float = 1.0,
    top_p: float | None = None,
    top_k: int | None = None,
    do_sample: bool = True,
) -> np.ndarray:
    """Sample tokens from logits with optional filtering and repetition penalty."""
    vocab_size = logits.shape[-1]

    if prev_tokens is not None and repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, prev_tokens, repetition_penalty)

    if not do_sample:
        return np.argmax(logits, axis=-1).astype(np.int64)

    original_shape = logits.shape
    flat = logits.reshape(-1, vocab_size)
    N = flat.shape[0]

    if top_k is not None and top_k > 0:
        k = min(top_k, vocab_size)
        top_idx = np.argpartition(flat, -k, axis=-1)[:, -k:]
        top_vals = np.take_along_axis(flat, top_idx, axis=-1)

        if top_p is not None and top_p < 1.0:
            top_vals = apply_top_p(top_vals, top_p)

        probs = softmax(top_vals)
        local = multinomial(probs)
        tokens = top_idx[np.arange(N), local]
    else:
        if top_p is not None and top_p < 1.0:
            flat = apply_top_p(flat, top_p)
        probs = softmax(flat)
        tokens = multinomial(probs)

    return tokens.reshape(original_shape[:-1]).astype(np.int64)
