"""Token-weighted cross-entropy on the trained positions only, in chunks.

Gemma 4's vocabulary has 262,144 entries; the full logits of a 32K-token row would take 17 GB in
bf16. Only trained tokens (typically 12-22% of a row) need logits, so the hidden states at those
positions are gathered first and the logits are computed chunk by chunk under activation
checkpointing (recomputed in backward, never stored). Gemma 4's final logit softcapping
(`tanh(x / 30) * 30`) is applied exactly as in the HF model.

Returned value is the *sum* Σ w·CE; normalization over the global batch happens in the trainer.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def trained_positions(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(row, column) of every position with a label; the prediction comes from column - 1."""
    rows, cols = torch.nonzero(labels != -100, as_tuple=True)
    if (cols == 0).any():
        raise ValueError("a trained token at position 0 has no preceding context")
    return rows, cols


def _chunk_loss(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, w: torch.Tensor,
                softcap: float | None) -> tuple[torch.Tensor, torch.Tensor]:
    logits = (h @ weight.t()).float()
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    ce = F.cross_entropy(logits, targets, reduction="none")
    return (ce * w).sum(), ce.detach()


def weighted_ce_sum(hidden: torch.Tensor, lm_weight: torch.Tensor, targets: torch.Tensor,
                    weights: torch.Tensor, softcap: float | None, chunk: int = 1024
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Σ weights·CE(softcap(hidden @ lm_weightᵀ), targets) and the per-token CE (detached).

    hidden: [N, H] states that predict `targets` [N]; weights: [N] float."""
    if hidden.shape[0] == 0:
        return hidden.sum() * 0.0, hidden.new_zeros(0, dtype=torch.float32)
    total = None
    ces = []
    for s in range(0, hidden.shape[0], chunk):
        h, t, w = hidden[s:s + chunk], targets[s:s + chunk], weights[s:s + chunk].float()
        if torch.is_grad_enabled():
            part, ce = checkpoint(_chunk_loss, h, lm_weight, t, w, softcap, use_reentrant=False)
        else:
            part, ce = _chunk_loss(h, lm_weight, t, w, softcap)
        total = part if total is None else total + part
        ces.append(ce)
    return total, torch.cat(ces)


def reference_weighted_ce_sum(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Plain implementation on full (already softcapped) logits [B, T, V], for tests."""
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = labels[:, 1:]
    mask = tgt != -100
    nll = -lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (nll * weights[:, 1:].float() * mask).sum()
