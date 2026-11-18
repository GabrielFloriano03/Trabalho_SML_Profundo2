"""Mede throughput (tokens/s) e pico de VRAM para dimensionar o cronograma.

Uso:
    python src/bench.py                 # varre micro-batches
    python src/bench.py --smoke         # smoke test: overfitar um mini-batch
"""

from __future__ import annotations

import argparse
import time

import torch

from model import SLM, SLMConfig


def human(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}T"


def build(cfg: SLMConfig, device: str):
    model = SLM(cfg).to(device)
    return model


def bench_microbatch(cfg, device, micro_bs, steps=12, warmup=4):
    """Roda alguns passos forward+backward e devolve (tokens/s, pico de VRAM em GB)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = build(cfg, device)
    opt = model.configure_optimizers(0.1, 3e-4, (0.9, 0.95), "cuda")

    x = torch.randint(0, cfg.real_vocab_size, (micro_bs, cfg.max_seq_len), device=device)
    y = torch.randint(0, cfg.real_vocab_size, (micro_bs, cfg.max_seq_len), device=device)

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with autocast:
            _, loss, _ = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    tokens = steps * micro_bs * cfg.max_seq_len
    peak = torch.cuda.max_memory_allocated() / 1e9

    del model, opt, x, y
    torch.cuda.empty_cache()
    return tokens / dt, peak


def smoke_test(cfg, device, micro_bs=2, steps=200):
    """O modelo deve conseguir overfitar um unico mini-batch ate a loss ~0."""
    print("\n=== SMOKE TEST: overfitar um mini-batch ===")
    model = build(cfg, device)
    opt = model.configure_optimizers(0.0, 1e-3, (0.9, 0.95), "cuda")
    x = torch.randint(0, cfg.real_vocab_size, (micro_bs, 256), device=device)
    y = torch.randint(0, cfg.real_vocab_size, (micro_bs, 256), device=device)

    first = last = None
    for step in range(steps):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss, _ = model(x, targets=y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step == 0:
            first = loss.item()
        if step % 40 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}")
        last = loss.item()

    print(f"  loss inicial {first:.4f} -> final {last:.4f}")
    print("  OK: o pipeline de treino consegue memorizar o batch."
          if last < 0.5 else
          "  ATENCAO: a loss nao caiu o suficiente, revisar o pipeline.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="roda o smoke test de overfit")
    ap.add_argument("--micro-bs", type=int, nargs="*", default=[2, 4, 6, 8, 12],
                    help="micro-batches a testar")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA indisponivel"
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = SLMConfig()
    model = SLM(cfg)
    total = model.num_params()
    non_emb = model.num_params(non_embedding=True)
    del model

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM {props.total_memory/1e9:.2f} GB | sm_{props.major}{props.minor}")
    print(f"Parametros treinaveis: {human(total)} (nao-embedding: {human(non_emb)})")
    print(f"Config: d_model={cfg.d_model} n_layer={cfg.n_layer} n_head={cfg.n_head} "
          f"n_kv_head={cfg.n_kv_head} ffn={cfg.ffn_hidden} ctx={cfg.max_seq_len}")
    print(f"Orcamento Chinchilla (20x params): {human(20 * total)} tokens\n")

    if args.smoke:
        smoke_test(cfg, device)
        return

    print(f"{'micro_bs':>9} {'tokens/s':>12} {'pico VRAM':>11}  {'1.1B tokens em':>16}")
    print("-" * 54)
    best = None
    for mbs in args.micro_bs:
        try:
            tps, peak = bench_microbatch(cfg, device, mbs)
        except torch.OutOfMemoryError:
            print(f"{mbs:>9} {'OOM':>12}")
            torch.cuda.empty_cache()
            continue
        hours = (20 * total) / tps / 3600
        print(f"{mbs:>9} {tps:>12,.0f} {peak:>9.2f} GB  {hours:>13.1f} h")
        if best is None or tps > best[1]:
            best = (mbs, tps, peak)

    if best:
        mbs, tps, peak = best
        print(f"\nMelhor: micro_bs={mbs} -> {tps:,.0f} tok/s, pico {peak:.2f} GB")


if __name__ == "__main__":
    main()
