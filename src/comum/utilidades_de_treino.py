"""Utilitarios compartilhados pelos tres estagios de treino."""

from __future__ import annotations

import csv
import json
import math
import os
import time

import torch

from arquitetura_modelo import SLM, SLMConfig


# --------------------------------------------------------------------------- #
# Learning rate schedules
# --------------------------------------------------------------------------- #
def lr_cosine(step: int, total_steps: int, max_lr: float, min_lr: float,
              warmup_steps: int) -> float:
    """Warmup linear seguido de decaimento cosseno ate min_lr."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def lr_wsd(step: int, total_steps: int, max_lr: float, min_lr: float,
           warmup_steps: int, decay_frac: float = 0.2) -> float:
    """Warmup-Stable-Decay: LR constante e queda linear no final."""
    decay_steps = int(total_steps * decay_frac)
    stable_end = total_steps - decay_steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step < stable_end:
        return max_lr
    if step >= total_steps:
        return min_lr
    progress = (step - stable_end) / max(1, decay_steps)
    return max_lr + (min_lr - max_lr) * progress


def get_lr(schedule: str, *args, **kwargs) -> float:
    return lr_wsd(*args, **kwargs) if schedule == "wsd" else lr_cosine(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class CSVLogger:
    """Grava as metricas em CSV para gerar as curvas de loss depois."""

    FIELDS = ["step", "tokens", "lr", "train_loss", "val_loss", "val_ppl",
              "pretrain_val_ppl", "grad_norm", "tokens_per_s", "elapsed_s"]

    def __init__(self, path: str, resume: bool = False):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        exists = os.path.exists(path)
        self.f = open(path, "a" if (resume and exists) else "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        if not (resume and exists):
            self.w.writeheader()
            self.f.flush()

    def log(self, **row):
        self.w.writerow({k: row.get(k, "") for k in self.FIELDS})
        self.f.flush()

    def close(self):
        self.f.close()


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, model: SLM, optimizer, step: int, tokens: int,
                    loader_state: dict, extra: dict | None = None,
                    weights_only_path: str | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model.cfg.to_dict(),
        "step": step,
        "tokens": tokens,
        "loader": loader_state,
        "extra": extra or {},
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)

    if weights_only_path:
        torch.save({"model": model.state_dict(), "config": model.cfg.to_dict()},
                   weights_only_path)


def load_model_from_checkpoint(path: str, device: str = "cuda",
                               map_config: dict | None = None) -> tuple[SLM, dict]:
    """Carrega so os pesos (usado para iniciar o estagio seguinte e para avaliar)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["config"])
    if map_config:
        cfg_dict.update(map_config)
    cfg = SLMConfig(**cfg_dict)
    model = SLM(cfg)
    state = ckpt["model"]
    # o weight tying faz lm_head.weight ser o mesmo tensor de tok_emb.weight
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"chaves inesperadas no checkpoint: {unexpected}")
    allowed = {"lm_head.weight"} if cfg.tie_embeddings else set()
    if set(missing) - allowed:
        raise RuntimeError(f"chaves faltando no checkpoint: {missing}")
    model.to(device)
    return model, ckpt


# --------------------------------------------------------------------------- #
# Avaliacao
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model: SLM, val_loader, max_batches: int = 40,
             amp_dtype=torch.bfloat16) -> tuple[float, float]:
    """Loss media e perplexidade no split de validacao."""
    was_training = model.training
    model.eval()
    total_loss, n = 0.0, 0
    for x, y in val_loader.batches(max_batches):
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _, loss, _ = model(x, targets=y)
        total_loss += loss.item()
        n += 1
    if was_training:
        model.train()
    mean = total_loss / max(1, n)
    return mean, math.exp(min(mean, 20.0))


def save_run_meta(path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def fmt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def human(n: float) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}T"
