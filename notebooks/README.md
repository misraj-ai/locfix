# GPU steps

Each script runs one step that needs a GPU, the competition environment or a model download.

| Script | Where | What |
| --- | --- | --- |
| `golden_render_check.py` | Colab or Kaggle with vLLM 0.19.1 | Compares our prompt token ids with vLLM's `/v1/chat/completions/render` at every trained turn of `tests/fixtures/records/s1_fixtures.jsonl` (or any records file). Must report `ok: true`. |
| `kaggle_lora_check.py` | Kaggle L4×4, competition model attached, vLLM 0.19.1 from the wheelhouse | Does vLLM load a LoRA adapter on the `-w4a16-ct` model when started like the harness? Run for E4B and 31B, `--mode harness` first, then `--mode text-only`. CPU self-test: `--selftest`. |

## Running `kaggle_lora_check.py` on Kaggle

1. New notebook, accelerator **GPU L4×4**, internet **off**; add the competition data, the
   wheelhouse dataset and the Gemma 4 `-w4a16-ct` model(s) as inputs.
2. Install vLLM 0.19.1 and its dependencies from the wheelhouse the same way the competition
   notebook does (`pip install --no-index --find-links <wheelhouse dir> ...`).
3. Add this script (upload or paste into a cell with `%%writefile kaggle_lora_check.py`), then:
   ```
   !python kaggle_lora_check.py --model-dir /kaggle/input/<...>/gemma-4-e4b-it-qat-w4a16-ct/1
   !python kaggle_lora_check.py --model-dir /kaggle/input/<...>/gemma-4-e4b-it-qat-w4a16-ct/1 --mode text-only
   !python kaggle_lora_check.py --model-dir /kaggle/input/<...>/gemma-4-31b-it-qat-w4a16-ct/1
   ```
4. Send back the `lora_check_*.json` files from `/kaggle/working/` (and the `vllm_*.log` files if
   the server did not start).

What the result means:
- `server_started: false` and `does_not_support_lora: true` in `harness` mode → adapters cannot be
  used with that model in the competition.
- `adapter_request_ok: true` and `adapter_changes_output: true` → adapters work.
- `adapter_changes_output: false` → vLLM accepted the adapter but ignored it (module names).

## Running it on Colab instead (A100 or H100)

The question (does vLLM 0.19.1 load a LoRA adapter on this checkpoint?) depends only on the vLLM
code and the checkpoint, so a single Colab GPU gives the same answer as Kaggle. Run as a shell
command (`!python ...`) or call `check(...)` from a cell; do not paste the file and run it as a cell.

```
!pip install -q vllm==0.19.1 transformers==5.13.1
!wget -q https://raw.githubusercontent.com/<you>/<repo>/main/notebooks/kaggle_lora_check.py   # or upload it

!python kaggle_lora_check.py --model-dir google/gemma-4-E4B-it-qat-w4a16-ct
!python kaggle_lora_check.py --model-dir google/gemma-4-E4B-it-qat-w4a16-ct --mode text-only
!python kaggle_lora_check.py --model-dir google/gemma-4-31B-it-qat-w4a16-ct     # ~17 GB download
```

or, in a Python cell after the file is in the working directory:

```python
from kaggle_lora_check import check
report = check("google/gemma-4-E4B-it-qat-w4a16-ct")
```

Reports are written to the working directory (`lora_check_<model>_<mode>.json`, `vllm_*.log`).
