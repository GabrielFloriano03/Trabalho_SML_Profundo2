"""Loop de treino compartilhado pelo mid-training e pelo SFT.

As duas etapas sao o mesmo loop com configuracoes diferentes:
  - mid-training: loss em todos os tokens, LR da ordem de 1e-4..3e-4
  - SFT        : loss mask (so tokens do assistente), LR da ordem de 1e-5..3e-5

Os dois partem de um checkpoint anterior carregando apenas os pesos do modelo,
sem o estado do otimizador, como pede o enunciado.
"""

from __future__ import annotations

import os
import time

import torch

from carregadores_de_dados import ShardLoader, ValLoader
from utilidades_de_treino import (CSVLogger, evaluate, fmt_time, get_lr, human,
                                  load_model_from_checkpoint, save_checkpoint,
                                  save_run_meta)


def run_stage(args, stage: str, use_loss_mask: bool):
    assert torch.cuda.is_available(), "este script precisa de GPU CUDA"
    device = "cuda"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # ---- modelo a partir do checkpoint anterior (so pesos) ----
    model, _ = load_model_from_checkpoint(args.init_from, device,
                                          map_config={"dropout": args.dropout})
    n_params = model.num_params()
    print(f"inicializado de {args.init_from} | {human(n_params)} parametros")

    optimizer = model.configure_optimizers(
        args.weight_decay, args.lr, (args.beta1, args.beta2), "cuda")

    # ---- dados ----
    grad_accum = args.global_batch_tokens // (args.micro_bs * args.seq_len)
    assert grad_accum >= 1, "global-batch-tokens menor que um micro-batch"
    real_global = grad_accum * args.micro_bs * args.seq_len

    train_loader = ShardLoader(
        args.data_dir, "train_[0-9]*.npy", args.micro_bs, args.seq_len, device,
        mask_pattern="train_mask_*.npy" if use_loss_mask else None,
    )
    total_tokens_disponiveis = train_loader.total_tokens
    budget = int(args.total_tokens) if args.total_tokens else int(
        total_tokens_disponiveis * args.epochs)
    total_steps = max(1, budget // real_global)

    val_loader = ValLoader(
        os.path.join(args.data_dir, "val.npy"), args.micro_bs, args.seq_len, device,
        mask_path=(os.path.join(args.data_dir, "val_mask.npy")
                   if use_loss_mask else None),
    )
    pretrain_val = None
    if args.pretrain_val and os.path.exists(args.pretrain_val):
        pretrain_val = ValLoader(args.pretrain_val, args.micro_bs, args.seq_len, device)

    # ---- resume ----
    ckpt_path = os.path.join(args.out_dir, f"{stage}_last.pt")
    start_step, tokens_seen, best_val = 0, 0, float("inf")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        tokens_seen = ckpt["tokens"]
        best_val = ckpt.get("extra", {}).get("best_val", float("inf"))
        train_loader.load_state_dict(ckpt["loader"])
        print(f"retomando do step {start_step} ({human(tokens_seen)} tokens)")

    logger = CSVLogger(args.log_file, resume=args.resume)

    print("=" * 72)
    print(f"Etapa                 : {stage} (loss mask: {'sim' if use_loss_mask else 'nao'})")
    print(f"Dados                 : {args.data_dir} ({human(total_tokens_disponiveis)} tokens)")
    print(f"Batch global          : {human(real_global)} tokens "
          f"({args.micro_bs} x {args.seq_len} x {grad_accum} acum)")
    print(f"Orcamento             : {human(budget)} tokens = {total_steps} steps "
          f"({budget / max(1, total_tokens_disponiveis):.2f} epocas)")
    print(f"LR                    : {args.lr:g} -> {args.min_lr:g} ({args.schedule}, "
          f"warmup {args.warmup_steps})")
    print("=" * 72)

    save_run_meta(os.path.join(args.out_dir, f"{stage}_meta.json"), {
        "stage": stage,
        "init_from": args.init_from,
        "config": model.cfg.to_dict(),
        "params": n_params,
        "global_batch_tokens": real_global,
        "total_steps": total_steps,
        "dataset_tokens": total_tokens_disponiveis,
        "args": vars(args),
    })

    # ---- loop ----
    model.train()
    t_start = time.time()
    t_last = t_start

    for step in range(start_step, total_steps):
        lr = get_lr(args.schedule, step, total_steps, args.lr, args.min_lr,
                    args.warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # a loss fica como tensor durante a acumulacao: um .item() por micro-step
        # forcaria uma sincronizacao de GPU por micro-batch
        loss_accum = torch.zeros((), device=device)
        for _ in range(grad_accum):
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
                  f"ETA {fmt_time(eta)}")
            logger.log(step=step + 1, tokens=tokens_seen, lr=lr,
                       train_loss=round(loss_accum, 5),
                       grad_norm=round(float(grad_norm), 4),
                       tokens_per_s=round(tps, 1),
                       elapsed_s=round(now - t_start, 1))

        if (step + 1) % args.eval_every == 0 or step + 1 == total_steps:
            val_loss, val_ppl = evaluate(model, val_loader, args.eval_batches)
            msg = f"  >> val loss {val_loss:.4f} | ppl {val_ppl:.2f}"
            row = dict(step=step + 1, tokens=tokens_seen, lr=lr,
                       val_loss=round(val_loss, 5), val_ppl=round(val_ppl, 3),
                       elapsed_s=round(time.time() - t_start, 1))
            if pretrain_val is not None:
                pv_loss, pv_ppl = evaluate(model, pretrain_val, args.eval_batches)
                msg += f" | ppl(fineweb) {pv_ppl:.2f}"
                row["pretrain_val_ppl"] = round(pv_ppl, 3)
            print(msg)
            logger.log(**row)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(os.path.join(args.out_dir, f"{stage}_best.pt"),
                                model, optimizer, step + 1, tokens_seen,
                                train_loader.state_dict(),
                                extra={"best_val": best_val, "val_ppl": val_ppl})

        if (step + 1) % args.ckpt_every == 0 or step + 1 == total_steps:
            save_checkpoint(ckpt_path, model, optimizer, step + 1, tokens_seen,
                            train_loader.state_dict(), extra={"best_val": best_val})
            print(f"  >> checkpoint salvo em {ckpt_path}")

        if args.max_hours and (time.time() - t_start) / 3600 >= args.max_hours:
            save_checkpoint(ckpt_path, model, optimizer, step + 1, tokens_seen,
                            train_loader.state_dict(), extra={"best_val": best_val})
            print(f"\nlimite de {args.max_hours}h atingido; use --resume para continuar.")
            break

    save_checkpoint(ckpt_path, model, optimizer, total_steps, tokens_seen,
                    train_loader.state_dict(), extra={"best_val": best_val},
                    weights_only_path=os.path.join(args.out_dir, f"{stage}_final.pt"))
    logger.close()
    print(f"\n{stage} concluido em {fmt_time(time.time() - t_start)} | "
          f"{human(tokens_seen)} tokens | melhor val loss {best_val:.4f}")


def add_common_args(ap, stage: str):
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--log-file", default=f"logs/{stage}.csv")
    ap.add_argument("--pretrain-val", default="data/pretrain/val.npy",
                    help="split de validacao do pre-treino, p/ comparar PPL entre etapas")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--global-batch-tokens", type=int, default=2 ** 17)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=30)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    return ap
