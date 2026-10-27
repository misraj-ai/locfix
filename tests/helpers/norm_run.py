"""One optimizer step on 2 rows, for the loss-normalization test (run directly or with torchrun).

    python tests/helpers/norm_run.py OUT_DIR RECORDS_A RECORDS_B TOKENIZER_DIR
    python -m torch.distributed.run --nproc_per_node 2 tests/helpers/norm_run.py ...
"""
import os
import zlib
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from tiny_gemma4 import tiny_model  # noqa: E402

from locfix.render.chat import load_tokenizer  # noqa: E402
from locfix.train.data import Source  # noqa: E402
from locfix.train.model import add_lora  # noqa: E402
from locfix.train.sft import TrainSettings, run_training  # noqa: E402


def settings(out, a, b, cache):
    return TrainSettings(output_dir=out, base_model="unused",
                         sources=[Source(a, weight=1.0), Source(b, weight=0.3)], epochs=1,
                         learning_rate=0.5, lora_r=4, lora_alpha=8, global_batch_rows=2, warmup_steps=0,
                         cache_dir=cache, bf16=False, export=False, optim="sgd", lr_scheduler="constant",
                         grad_clip=1e9, seed=7)


def init_model():
    torch.manual_seed(0)
    m = tiny_model("31b", vocab=262144)
    return m


if __name__ == "__main__":
    out, a, b, tok_dir = sys.argv[1:5]
    tok = load_tokenizer(tok_dir)
    m = init_model()
    torch.manual_seed(1)
    orig_add = add_lora

    def seeded_add(model, *x, **k):
        torch.manual_seed(1)
        pm = orig_add(model, *x, **k)
        for n, p in pm.named_parameters():          # non-zero B so A also gets a gradient
            if "lora_B" in n:
                p.data.normal_(0, 0.02, generator=torch.Generator().manual_seed(zlib.crc32(n.encode())))
        return pm

    import locfix.train.model as lm
    lm.add_lora = seeded_add
    run_training(settings(out, a, b, os.path.join(out, "..", "cache")), use_cpu=True, model=m, tokenizer=tok)
