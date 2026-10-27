"""`locfix train`: one LoRA run on record files (every stage uses this; runs differ in data only).

    accelerate launch --config_file configs/accelerate/ddp_8gpu.yaml -m locfix.cli train \
        --config configs/runs/S1-e4b.yaml

Loss: Σ w·CE over the trained tokens of the *global* batch (all processes and accumulation
steps) divided by Σ w over the same tokens. Implemented by counting Σ w of the upcoming
accumulation steps on every process (`_get_num_items_in_batch`), all-reducing it, dividing each
micro-batch's Σ w·CE by it, and multiplying by the number of processes (DDP averages gradients).
Checked by `tests/test_train.py` (2 processes × 1 step == 1 process × 2 accumulation steps).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .data import RowDataset, Source, collate_rows, load_sources, make_plan, plan_summary


# --------------------------------------------------------------------------- config
@dataclass
class TrainSettings:
    output_dir: str
    base_model: str                       # dequantized text-only checkpoint
    sources: list[Source]
    val_sources: list[Source] = field(default_factory=list)
    tokenizer: str | None = None          # default: base_model
    role: str = "agent"
    model_key: str = "e4b"
    epochs: int = 3
    learning_rate: float = 2e-4
    lora_r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    init_adapter: str | None = None       # warm start (EM rounds, Stage 5)
    global_batch_rows: int = 16
    per_device_rows: int = 1
    max_seq_len: int = 32768
    packing: bool = False
    length_grouping: bool = False
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    ce_chunk: int = 1024
    warmup_steps: int = 10
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0
    seed: int = 1234
    cache_dir: str | None = None
    bf16: bool = True
    logging_steps: int = 1
    report_to: list[str] = field(default_factory=list)
    export: bool = True
    serve_id: str | None = None
    optim: str = "adamw_torch"
    lr_scheduler: str = "cosine_with_min_lr"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> TrainSettings:
        t = cfg["train"]
        key = t.get("model", "e4b")
        m = cfg["models"][key]
        lr = t.get("learning_rate") or t["lr"][key]
        lr *= float(t.get("lr_scale", 1.0))
        srcs = [Source(**s) for s in t.get("sources", [])]
        if not srcs:
            raise ValueError("train.sources is empty: list the record files to train on")
        return cls(
            output_dir=t["output_dir"], base_model=t.get("base_model") or m["dequantized"], sources=srcs,
            val_sources=[Source(**s) for s in t.get("val_sources", [])], tokenizer=t.get("tokenizer"),
            role=t.get("role", "agent"), model_key=key, epochs=int(t["epochs"]), learning_rate=float(lr),
            lora_r=int(t["lora"]["r"]), lora_alpha=int(t["lora"]["alpha"]),
            lora_dropout=float(t["lora"].get("dropout", 0.0)), init_adapter=t.get("init_adapter"),
            global_batch_rows=int(t["global_batch_sequences"]), per_device_rows=int(t.get("per_device_batch", 1)),
            max_seq_len=int(t["max_seq_len"]), packing=bool(t.get("packing", False)),
            length_grouping=bool(t.get("length_grouping", False)),
            attn_implementation=t.get("attn_implementation", "sdpa"),
            gradient_checkpointing=bool(t.get("gradient_checkpointing", True)), ce_chunk=int(t.get("ce_chunk", 1024)),
            warmup_steps=int(t["schedule"]["warmup_steps"]), min_lr_ratio=float(t["schedule"]["min_lr_ratio"]),
            weight_decay=float(t["optimizer"]["weight_decay"]), betas=tuple(t["optimizer"]["betas"]),
            grad_clip=float(t["optimizer"]["grad_clip"]), seed=int(cfg.get("seed", 1234)),
            cache_dir=t.get("cache_dir"), bf16=bool(t.get("bf16", True)),
            logging_steps=int(t.get("logging_steps", 1)), report_to=list(t.get("report_to", [])),
            export=bool(t.get("export", True)), serve_id=m.get("serve_id"),
            optim=t["optimizer"].get("hf_name", "adamw_torch"),
        )

    def grad_accum(self, world_size: int) -> int:
        per_step = world_size * self.per_device_rows
        if self.global_batch_rows % per_step:
            raise ValueError(f"global batch {self.global_batch_rows} is not a multiple of "
                             f"{world_size} processes × {self.per_device_rows} rows")
        return self.global_batch_rows // per_step


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


# --------------------------------------------------------------------------- trainer
def make_trainer_class():
    from transformers import Trainer, TrainerCallback

    class LocfixTrainer(Trainer):
        """Token-weighted global loss, our row order, adapter-only saving, per-source logging."""

        def __init__(self, *a, source_labels: list[str], epoch_end_steps: list[int], settings: TrainSettings,
                     val_data: tuple | None = None, **kw):
            super().__init__(*a, **kw)
            self.model_accepts_loss_kwargs = True       # we normalize by num_items_in_batch ourselves
            self.source_labels = source_labels
            self.settings = settings
            self.val_data = val_data
            n = len(source_labels)
            self._src_num = torch.zeros(n, dtype=torch.float64)
            self._src_den = torch.zeros(n, dtype=torch.float64)
            self._tokens = 0
            self._t_last = time.time()
            self.add_callback(EpochCallback(self, epoch_end_steps))

        def _get_train_sampler(self, train_dataset=None):
            return torch.utils.data.SequentialSampler(train_dataset or self.train_dataset)

        def _get_num_items_in_batch(self, batch_samples, device):
            total = sum(b["loss_weights"].sum() for b in batch_samples)
            total = torch.as_tensor(total, dtype=torch.float32, device=device)
            if self.accelerator.num_processes > 1:
                total = self.accelerator.reduce(total, reduction="sum")
            return total

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            out = model(**inputs)
            if num_items_in_batch is None:                # evaluation path: local normalization
                num_items_in_batch = inputs["loss_weights"].sum()
            loss = out["loss"] / num_items_in_batch.clamp(min=1e-8) * self.accelerator.num_processes
            if os.environ.get("LOCFIX_DEBUG_LOSS"):
                print(f"[rank {self.accelerator.process_index}] n={inputs['input_ids'].shape[1]} "
                      f"sum={float(out['loss']):.4f} w={float(out['weights'].sum()):.2f} "
                      f"denom={float(num_items_in_batch):.2f} loss={float(loss):.6f}", flush=True)
            if model.training and "source_ids" in out:
                with torch.no_grad():
                    sid = out["source_ids"].long().cpu()
                    self._src_num.index_add_(0, sid, (out["ce"] * out["weights"]).double().cpu())
                    self._src_den.index_add_(0, sid, out["weights"].double().cpu())
                self._tokens += int(inputs["input_ids"].numel())
            return (loss, out) if return_outputs else loss

        def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
            if "loss" in logs and self.model.training:
                num, den = self._src_num.clone(), self._src_den.clone()
                if self.accelerator.num_processes > 1:
                    num = self.accelerator.reduce(num.to(self.args.device), reduction="sum").cpu()
                    den = self.accelerator.reduce(den.to(self.args.device), reduction="sum").cpu()
                for i, lab in enumerate(self.source_labels):
                    if den[i] > 0:
                        logs[f"loss/{lab}"] = round(float(num[i] / den[i]), 5)
                now = time.time()
                logs["tokens_per_s_per_device"] = round(self._tokens / max(now - self._t_last, 1e-6), 1)
                self._src_num.zero_()
                self._src_den.zero_()
                self._tokens, self._t_last = 0, now
            super().log(logs, start_time)

        # ---- saving: adapters only (the base never changes) -------------------------------
        def _save(self, output_dir: str | None = None, state_dict=None) -> None:
            self.save_adapter(output_dir or self.args.output_dir)

        def save_adapter(self, output_dir: str) -> None:
            sd = self.accelerator.get_state_dict(self.model)
            if self.accelerator.is_main_process:
                lm = self.accelerator.unwrap_model(self.model)
                peft_sd = {k[len("peft."):]: v for k, v in sd.items() if k.startswith("peft.")}
                lm.peft.save_pretrained(output_dir, state_dict=peft_sd, safe_serialization=True)
            self.accelerator.wait_for_everyone()

        # ---- held-out NLL per stage ---------------------------------------------------------
        @torch.no_grad()
        def evaluate_val(self) -> dict[str, float]:
            if not self.val_data:
                return {}
            examples, stages = self.val_data
            model = self.model
            was_training = model.training
            model.eval()
            n = len(stages)
            num = torch.zeros(n, dtype=torch.float64, device=self.args.device)
            den = torch.zeros(n, dtype=torch.float64, device=self.args.device)
            rank, world = self.accelerator.process_index, self.accelerator.num_processes
            ds = RowDataset([[i] for i in range(len(examples))], examples, [Source("val")] * 1)
            for k in range(rank, len(ds), world):
                batch = {key: v.unsqueeze(0).to(self.args.device) for key, v in ds[k].items()}
                batch.pop("source_ids")
                with self.compute_loss_context_manager():
                    out = model(**batch)
                s = stages.index(examples[k].stage)
                num[s] += float((out["ce"] * out["weights"]).sum())
                den[s] += float(out["weights"].sum())
            if world > 1:
                num = self.accelerator.reduce(num, reduction="sum")
                den = self.accelerator.reduce(den, reduction="sum")
            if was_training:
                model.train()
            res = {f"val_nll/{st}": round(float(num[i] / den[i]), 5) for i, st in enumerate(stages) if den[i] > 0}
            tot = float(den.sum())
            if tot > 0:
                res["val_nll/all"] = round(float(num.sum() / tot), 5)
            return res

    class EpochCallback(TrainerCallback):
        def __init__(self, trainer, epoch_end_steps: list[int]):
            self.trainer, self.ends = trainer, epoch_end_steps

        def on_step_end(self, args, state, control, **kw):
            if state.global_step in self.ends:
                ep = self.ends.index(state.global_step) + 1
                tr = self.trainer
                tr.save_adapter(os.path.join(args.output_dir, f"epoch-{ep}"))
                val = tr.evaluate_val()
                if val:
                    tr.log({**val, "epoch_done": ep})
                    if tr.accelerator.is_main_process:
                        with open(os.path.join(args.output_dir, "val.jsonl"), "a") as f:
                            f.write(json.dumps({"epoch": ep, "step": state.global_step, **val}) + "\n")
            return control

    return LocfixTrainer


# --------------------------------------------------------------------------- run
def build_data(s: TrainSettings, tok, ws: int, main_first) -> tuple:
    with main_first():
        examples, reports = load_sources(s.sources, tok, s.max_seq_len, s.cache_dir)
        val_examples, val_reports = load_sources(s.val_sources, tok, s.max_seq_len, s.cache_dir) \
            if s.val_sources else ([], [])
    ga = s.grad_accum(ws)
    plan = make_plan(examples, s.sources, s.epochs, ws * ga * s.per_device_rows, s.max_seq_len,
                     s.packing, s.seed, s.length_grouping)
    if plan.steps == 0:
        raise ValueError(f"no optimizer step: {len(examples)} examples, {ws * ga} rows per step")
    return examples, reports, val_examples, val_reports, plan, ga


def run_training(s: TrainSettings, cfg: dict[str, Any] | None = None, use_cpu: bool = False,
                 model=None, tokenizer=None) -> dict[str, Any]:
    """Train one adapter. `model`/`tokenizer` may be passed (tests); otherwise loaded from
    `s.base_model`."""
    from transformers import AutoTokenizer, TrainingArguments

    from ..runmeta import write_run_meta
    from .model import LossModel, add_lora, load_base

    from accelerate import PartialState

    ws = world_size()
    state = PartialState(cpu=use_cpu)
    tok = tokenizer or AutoTokenizer.from_pretrained(s.tokenizer or s.base_model)
    examples, reports, val_examples, val_reports, plan, ga = build_data(s, tok, ws, state.main_process_first)
    summary = plan_summary(plan, examples, s.sources)
    sched_kwargs = {"min_lr_rate": s.min_lr_ratio} if s.lr_scheduler == "cosine_with_min_lr" else {}
    args = TrainingArguments(
        output_dir=s.output_dir, per_device_train_batch_size=s.per_device_rows,
        gradient_accumulation_steps=ga, learning_rate=s.learning_rate, max_steps=plan.steps,
        lr_scheduler_type=s.lr_scheduler, lr_scheduler_kwargs=sched_kwargs, optim=s.optim,
        warmup_steps=s.warmup_steps, weight_decay=s.weight_decay, adam_beta1=s.betas[0],
        adam_beta2=s.betas[1], max_grad_norm=s.grad_clip, bf16=s.bf16 and not use_cpu, logging_steps=s.logging_steps,
        save_strategy="no", report_to=s.report_to, remove_unused_columns=False, seed=s.seed,
        data_seed=s.seed, average_tokens_across_devices=True, ddp_find_unused_parameters=False,
        dataloader_num_workers=0 if use_cpu else 2, use_cpu=use_cpu, gradient_checkpointing=False,
    )

    if model is None:
        model = load_base(s.base_model, attn_implementation=s.attn_implementation,
                          gradient_checkpointing=s.gradient_checkpointing)
    peft_model = add_lora(model, s.lora_r, s.lora_alpha, s.lora_dropout, s.init_adapter)
    loss_model = LossModel(peft_model, s.ce_chunk)

    Trainer = make_trainer_class()
    stages = sorted({e.stage for e in val_examples})
    trainer = Trainer(model=loss_model, args=args, train_dataset=RowDataset(plan.rows, examples, s.sources),
                      data_collator=collate_rows, source_labels=[x.label for x in s.sources],
                      epoch_end_steps=plan.epoch_end_steps(), settings=s,
                      val_data=(val_examples, stages) if val_examples else None)
    if trainer.accelerator.is_main_process:
        Path(s.output_dir).mkdir(parents=True, exist_ok=True)
        trainable = sum(p.numel() for p in loss_model.parameters() if p.requires_grad)
        write_run_meta(s.output_dir, cfg or {}, {
            "settings": {k: (v if not isinstance(v, list) or not v or not isinstance(v[0], Source)
                             else [x.__dict__ for x in v]) for k, v in s.__dict__.items()},
            "world_size": ws, "gradient_accumulation_steps": ga, "plan": summary,
            "sources": [r.__dict__ | {"errors": r.errors[:20]} for r in reports],
            "val_sources": [r.__dict__ | {"errors": r.errors[:20]} for r in val_reports],
            "trainable_parameters": trainable,
        })
        print(json.dumps({"plan": summary, "trainable_parameters": trainable}, indent=2))
    result = trainer.train()
    final = os.path.join(s.output_dir, "final")
    trainer.save_adapter(final)
    out = {"train": result.metrics, "plan": summary, "final_adapter": final}
    if s.export and trainer.accelerator.is_main_process:
        from .export import export_adapter

        base_cfg = loss_model.causal_lm.config
        out["exports"] = {}
        for d in sorted(Path(s.output_dir).glob("epoch-*")) + [Path(final)]:
            rep = export_adapter(d, d.parent / "export" / d.name, base_cfg, serve_id=s.serve_id)
            out["exports"][d.name] = rep
        (Path(s.output_dir) / "result.json").write_text(json.dumps(out, indent=2, default=str))
    return out


# --------------------------------------------------------------------------- CLI
def add_args_cmd_train(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cpu", action="store_true", help="run on CPU (tests only)")


def cmd_train(args: argparse.Namespace, cfg: dict) -> int:
    s = TrainSettings.from_config(cfg)
    run_training(s, cfg, use_cpu=args.cpu)
    return 0


def add_args_cmd_plan(p: argparse.ArgumentParser) -> None:
    p.add_argument("--world-size", type=int, default=8, help="number of GPUs the run will use")
    p.add_argument("--tokenizer", help="tokenizer folder (default: train.tokenizer or the base model)")


def cmd_plan(args: argparse.Namespace, cfg: dict) -> int:
    """Dry run: tokenize the sources and print the plan (steps, tokens, visits per task). No GPU."""
    from contextlib import nullcontext

    from transformers import AutoTokenizer

    s = TrainSettings.from_config(cfg)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or s.tokenizer or s.base_model)
    examples, reports, _, _, plan, ga = build_data(s, tok, args.world_size, nullcontext)
    print(json.dumps({"world_size": args.world_size, "gradient_accumulation_steps": ga,
                      "learning_rate": s.learning_rate, "plan": plan_summary(plan, examples, s.sources),
                      "sources": [{k: v for k, v in r.__dict__.items() if k != "errors"} for r in reports]},
                     indent=2))
    return 0
