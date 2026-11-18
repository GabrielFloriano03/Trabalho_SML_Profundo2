"""Etapa 1: pre-treino (next-token prediction) do SLM.

Uma GPU, bf16 com autocast, gradient accumulation para atingir o batch global
alvo mesmo com 6 GB de VRAM, e checkpoints retomaveis.

Uso tipico:
    python src/etapa1_pretreino/treinar_pretreino.py                   # usa os defaults abaixo
    python src/etapa1_pretreino/treinar_pretreino.py --resume          # continua de checkpoints/pretrain_last.pt
    python src/etapa1_pretreino/treinar_pretreino.py --max-hours 6     # para de forma limpa depois de 6h
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# os modulos compartilhados entre as etapas ficam em src/comum/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comum"))

from arquitetura_modelo import SLM, SLMConfig
from carregadores_de_dados import ShardLoader, ValLoader
from utilidades_de_treino import (CSVLogger, evaluate, fmt_time, get_lr, human,
                                  save_checkpoint, save_run_meta)


def parse_args():
    ap = argparse.ArgumentParser()
    # dados
    ap.add_argument("--data-dir", default="data/pretrain")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--log-file", default="logs/pretrain.csv")
    # arquitetura
    ap.add_argument("--n-layer", type=int, default=10)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-kv-head", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--ffn-hidden", type=int, default=1408)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0)
    # orcamento
    ap.add_argument("--total-tokens", type=float, default=1.1e9,
                    help="orcamento de tokens (Chinchilla: ~20x os parametros)")
    ap.add_argument("--global-batch-tokens", type=int, default=2 ** 18,
                    help="tokens por passo de otimizacao")
    ap.add_argument("--micro-bs", type=int, default=8,
                    help="sequencias por forward (limitado pela VRAM)")
    # otimizacao
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    # rotina
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compile", action="store_true", help="tenta torch.compile")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "este script precisa de GPU CUDA"
    device = "cuda"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # ---------------- modelo ----------------
    cfg = SLMConfig(
        n_layer=args.n_layer, n_head=args.n_head, n_kv_head=args.n_kv_head,
        d_model=args.d_model, ffn_hidden=args.ffn_hidden,
        max_seq_len=args.seq_len, dropout=args.dropout,
    )
    model = SLM(cfg).to(device)
    n_params = model.num_params()
    n_params_ne = model.num_params(non_embedding=True)

    optimizer = model.configure_optimizers(
        args.weight_decay, args.lr, (args.beta1, args.beta2), "cuda")

    raw_model = model
    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile ativo")
        except Exception as exc:  # pragma: no cover - depende do ambiente
            print(f"torch.compile indisponivel ({exc}); seguindo em eager")

    # ---------------- dados ----------------
    grad_accum = args.global_batch_tokens // (args.micro_bs * args.seq_len)
    assert grad_accum >= 1, "global-batch-tokens menor que um micro-batch"
    real_global = grad_accum * args.micro_bs * args.seq_len
    total_steps = int(args.total_tokens) // real_global

    train_loader = ShardLoader(args.data_dir, "train_*.npy", args.micro_bs,
                               args.seq_len, device)
    val_loader = ValLoader(os.path.join(args.data_dir, "val.npy"),
                           args.micro_bs, args.seq_len, device)

    # ---------------- resume ----------------
    ckpt_path = os.path.join(args.out_dir, "pretrain_last.pt")
    start_step, tokens_seen, best_val = 0, 0, float("inf")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        tokens_seen = ckpt["tokens"]
        best_val = ckpt.get("extra", {}).get("best_val", float("inf"))
        train_loader.load_state_dict(ckpt["loader"])
        print(f"retomando do step {start_step} ({human(tokens_seen)} tokens)")

    logger = CSVLogger(args.log_file, resume=args.resume)

    print("=" * 72)
    print(f"Parametros treinaveis : {human(n_params)} (nao-embedding {human(n_params_ne)})")
    print(f"Batch global          : {human(real_global)} tokens "
          f"({args.micro_bs} x {args.seq_len} x {grad_accum} acum)")
    print(f"Orcamento             : {human(args.total_tokens)} tokens "
          f"= {total_steps} steps ({20 * n_params / args.total_tokens:.1f}x Chinchilla)")
    print(f"LR                    : {args.lr:g} -> {args.min_lr:g} ({args.schedule}, "
          f"warmup {args.warmup_steps})")
    print("=" * 72)

    save_run_meta(os.path.join(args.out_dir, "pretrain_meta.json"), {
        "stage": "pretrain",
        "config": cfg.to_dict(),
        "params": n_params,
        "params_non_embedding": n_params_ne,
        "global_batch_tokens": real_global,
        "total_steps": total_steps,
        "args": vars(args),
    })

    # ---------------- loop ----------------
    model.train()
    t_start = time.time()
    t_last = t_start

    for step in range(start_step, total_steps):
        lr = get_lr(args.schedule, step, total_steps, args.lr, args.min_lr,
                    args.warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # a loss fica como tensor durante a acumulacao: um .item() por micro-step
        # forcaria 64 sincronizacoes de GPU por passo de otimizacao
        loss_accum = torch.zeros((), device=device)
        for micro in range(grad_accum):
            x, y = train_loader.next_batch()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, targets=y)
            loss = loss / grad_accum
            loss_accum += loss.detach()
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen += real_global

        if (step + 1) % args.log_every == 0:
            torch.cuda.synchronize()
            now = time.time()
            loss_accum = loss_accum.item()
            tps = args.log_every * real_global / (now - t_last)
            t_last = now
            eta = (total_steps - step - 1) * real_global / max(tps, 1.0)
            print(f"step {step+1:>6}/{total_steps} | loss {loss_accum:.4f} | "
                  f"lr {lr:.2e} | gn {grad_norm:.2f} | {tps:,.0f} tok/s | "
                  f"{human(tokens_seen)} tok | ETA {fmt_time(eta)}")
            logger.log(step=step + 1, tokens=tokens_seen, lr=lr,
                       train_loss=round(loss_accum, 5),
                       grad_norm=round(float(grad_norm), 4),
                       tokens_per_s=round(tps, 1),
                       elapsed_s=round(now - t_start, 1))

        if (step + 1) % args.eval_every == 0 or step + 1 == total_steps:
            val_loss, val_ppl = evaluate(raw_model, val_loader, args.eval_batches)
            print(f"  >> val loss {val_loss:.4f} | perplexidade {val_ppl:.2f}")
            logger.log(step=step + 1, tokens=tokens_seen, lr=lr,
                       val_loss=round(val_loss, 5), val_ppl=round(val_ppl, 3),
                       elapsed_s=round(time.time() - t_start, 1))
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(os.path.join(args.out_dir, "pretrain_best.pt"),
                                raw_model, optimizer, step + 1, tokens_seen,
                                train_loader.state_dict(),
                                extra={"best_val": best_val, "val_ppl": val_ppl})

        if (step + 1) % args.ckpt_every == 0 or step + 1 == total_steps:
            save_checkpoint(ckpt_path, raw_model, optimizer, step + 1, tokens_seen,
                            train_loader.state_dict(),
                            extra={"best_val": best_val})
            print(f"  >> checkpoint salvo em {ckpt_path}")

        if args.max_hours and (time.time() - t_start) / 3600 >= args.max_hours:
            save_checkpoint(ckpt_path, raw_model, optimizer, step + 1, tokens_seen,
                            train_loader.state_dict(), extra={"best_val": best_val})
            print(f"\nlimite de {args.max_hours}h atingido; checkpoint salvo. "
                  f"Use --resume para continuar.")
            break

    # pesos finais, sem estado do otimizador (entrada do mid-training)
    save_checkpoint(ckpt_path, raw_model, optimizer, total_steps, tokens_seen,
                    train_loader.state_dict(), extra={"best_val": best_val},
                    weights_only_path=os.path.join(args.out_dir, "pretrain_final.pt"))
    logger.close()
    print(f"\nconcluido em {fmt_time(time.time() - t_start)} | "
          f"{human(tokens_seen)} tokens | melhor val loss {best_val:.4f}")


if __name__ == "__main__":
    main()
