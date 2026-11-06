"""Check that rotation with all quantization off leaves the model's logits unchanged."""
import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_wikitext2
from src.hadamard import self_test
from src.wrapper import QuantConfig, wrap_model


@torch.no_grad()
def get_logits(model, windows):
    return torch.cat([model(input_ids=w[None].cuda(), use_cache=False).logits.float().cpu() for w in windows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--block", default="128")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    block = "full" if args.block == "full" else int(args.block)

    self_test()
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.model)
    windows = load_wikitext2(tok, seq_len=512, max_windows=2)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, device_map="cuda")
    ref = get_logits(model, windows)
    cfg = QuantConfig(w_bits=None, a_bits=None, out_bits=None, rotate_in=True, rotate_out=True,
                      block_in=block, block_out=block, seed=args.seed, collect_stats=False)
    wrap_model(model, cfg)
    new = get_logits(model, windows)
    diff = (new - ref).abs()
    max_d, mean_d = diff.max().item(), diff.mean().item()
    print(f"max abs diff {max_d:.3e} | mean abs diff {mean_d:.3e} | rel diff "
          f"{(diff.norm() / ref.norm()).item():.3e}")
    ok = max_d < 1e-3
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
