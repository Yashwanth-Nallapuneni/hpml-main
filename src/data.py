"""WikiText-2 loading for perplexity evaluation."""
from typing import Optional

import torch
from datasets import load_dataset


def load_wikitext2(tokenizer, split: str = "test", seq_len: int = 2048,
                   max_windows: Optional[int] = None) -> torch.LongTensor:
    """Tokenize the whole split once; return non-overlapping windows [n, seq_len]."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    n = ids.numel() // seq_len
    windows = ids[: n * seq_len].reshape(n, seq_len)
    if max_windows is not None:
        windows = windows[:max_windows]
    return windows


def load_c4(tokenizer, seq_len: int = 2048, max_windows: Optional[int] = 256, split: str = "validation") -> torch.LongTensor:
    """C4 as in GPTQ: first 1100 documents of shard 0 of the split joined by spaces, non-overlapping windows."""
    f = {"validation": "en/c4-validation.00000-of-00008.json.gz", "train": "en/c4-train.00000-of-01024.json.gz"}[split]
    ds = load_dataset("allenai/c4", data_files={split: f}, split=split)
    ids = tokenizer(" ".join(ds[:1100]["text"]), return_tensors="pt").input_ids[0]
    n = ids.numel() // seq_len
    windows = ids[: n * seq_len].reshape(n, seq_len)
    return windows if max_windows is None else windows[:max_windows]


PILE = {"pilecc": "Pile-CC", "owt2": "OpenWebText2", "stackex": "StackExchange", "github": "Github", "wiki": "Wikipedia (en)",
        "uspto": "USPTO Backgrounds", "pmc": "PubMed Central", "freelaw": "FreeLaw", "arxiv": "ArXiv", "pubmed": "PubMed Abstracts"}


def load_pile(tokenizer, domain: str, part: str, seq_len: int = 2048, max_windows: Optional[int] = None) -> torch.LongTensor:
    """One Pile domain from NeelNanda/pile-10k; part 'calib' takes even-indexed docs, 'eval' odd-indexed, so they never share a doc."""
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    docs = [t for t, m in zip(ds["text"], ds["meta"]) if m["pile_set_name"] == PILE[domain]]
    ids = tokenizer("\n\n".join(docs[{"calib": 0, "eval": 1}[part]::2]), return_tensors="pt").input_ids[0]
    n = ids.numel() // seq_len
    windows = ids[: n * seq_len].reshape(n, seq_len)
    return windows if max_windows is None else windows[:max_windows]


def load_text(tokenizer, name: str, calib: bool, seq_len: int = 2048, max_windows: Optional[int] = None) -> torch.LongTensor:
    """Calibration (train side) or evaluation (test side) windows for wikitext, c4 or a PILE domain key."""
    if name == "wikitext":
        return load_wikitext2(tokenizer, split="train" if calib else "test", seq_len=seq_len, max_windows=max_windows)
    if name == "c4":
        return load_c4(tokenizer, seq_len=seq_len, max_windows=max_windows, split="train" if calib else "validation")
    return load_pile(tokenizer, name, "calib" if calib else "eval", seq_len, max_windows)
