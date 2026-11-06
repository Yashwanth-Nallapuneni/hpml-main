"""Perplexity and next-token accuracy evaluation."""
import math
from typing import Dict

import torch


@torch.no_grad()
def evaluate_lm(model, windows: torch.LongTensor, batch_size: int = 1) -> Dict[str, float]:
    """One pass over all windows. Returns
    ppl            : exp of the token-weighted mean causal-LM NLL
    next_token_acc : fraction of positions where argmax(logits[t]) == token[t+1]
    Both use the same 2047 predicted positions per 2048-token window."""
    model.eval()
    total_nll, total_tok, total_correct = 0.0, 0, 0
    n = windows.shape[0]
    for i in range(0, n, batch_size):
        batch = windows[i : i + batch_size].to(model.device)
        out = model(input_ids=batch, labels=batch, use_cache=False)
        ntok = batch.shape[0] * (batch.shape[1] - 1)  # model shifts labels internally
        total_nll += out.loss.float().item() * ntok
        pred = out.logits[:, :-1].argmax(dim=-1)  # prediction for position t+1, made at t
        total_correct += (pred == batch[:, 1:]).sum().item()
        total_tok += ntok
        done = i // batch_size + 1
        if done % 10 == 0:
            print(f"  [{min(i + batch_size, n)}/{n}] running ppl {math.exp(total_nll / total_tok):.3f}"
                  f" | next-token acc {total_correct / total_tok:.4f}", flush=True)
    return {"ppl": math.exp(total_nll / total_tok), "next_token_acc": total_correct / total_tok}


def perplexity(model, windows: torch.LongTensor, batch_size: int = 1) -> float:
    """Backward-compatible wrapper: perplexity only."""
    return evaluate_lm(model, windows, batch_size)["ppl"]
