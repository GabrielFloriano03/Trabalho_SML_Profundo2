"""Etapa 4: avaliacao do SLM.

Implementa as duas metricas obrigatorias do enunciado:

  1. Perplexidade no split de validacao do corpus de pre-treino (dados que o
     modelo nunca viu).
  2. Benchmarks de multipla escolha por log-likelihood: para cada alternativa
     calcula-se a log-verossimilhanca da continuacao dado o contexto e escolhe-se
     a maior. Nao ha geracao autorregressiva envolvida.

Alem da acuracia bruta reportamos acc_norm (log-verossimilhanca dividida pelo
numero de caracteres da alternativa), que e o padrao em HellaSwag/ARC e corrige
o vies a favor de alternativas curtas.

Uso:
    python src/evaluate.py --ckpt checkpoints/sft_final.pt
    python src/evaluate.py --ckpt checkpoints/pretrain_final.pt --tasks hellaswag,arc_easy
    python src/evaluate.py --ckpt ... --limit 500       # execucao rapida
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re

import torch
import torch.nn.functional as F
from datasets import load_dataset

import tiktoken
from data import ValLoader
from train_utils import load_model_from_checkpoint

ENC = tiktoken.get_encoding("gpt2")


# --------------------------------------------------------------------------- #
# Perplexidade
# --------------------------------------------------------------------------- #
@torch.no_grad()
def perplexity(model, val_path: str, seq_len: int, batch_size: int,
               max_batches: int | None, device: str) -> tuple[float, float]:
    loader = ValLoader(val_path, batch_size, seq_len, device)
    total, n = 0.0, 0
    for x, y in loader.batches(max_batches):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss, _ = model(x, targets=y)
        total += loss.item()
        n += 1
    mean = total / max(1, n)
    return mean, math.exp(mean)


# --------------------------------------------------------------------------- #
# Tarefas de multipla escolha
# --------------------------------------------------------------------------- #
def _clean_hellaswag(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_hellaswag(limit=None):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        ctx = row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize()
        yield {
            "context": _clean_hellaswag(ctx),
            "choices": [" " + _clean_hellaswag(e) for e in row["endings"]],
            "label": int(row["label"]),
        }


def load_arc(subset: str, limit=None):
    ds = load_dataset("allenai/ai2_arc", subset, split="test")
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        labels = row["choices"]["label"]
        if row["answerKey"] not in labels:
            continue
        yield {
            "context": "Question: " + row["question"] + "\nAnswer:",
            "choices": [" " + t for t in row["choices"]["text"]],
            "label": labels.index(row["answerKey"]),
        }


def load_piqa(limit=None):
    # ybisk/piqa ainda e distribuido como script de carregamento, que versoes
    # recentes de `datasets` nao executam mais. A conversao parquet automatica
    # do Hub (refs/convert/parquet) tem os mesmos dados, sem script.
    try:
        ds = load_dataset("ybisk/piqa", split="validation")
    except Exception:
        ds = load_dataset("ybisk/piqa", split="validation",
                          revision="refs/convert/parquet")
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        yield {
            "context": "Question: " + row["goal"] + "\nAnswer:",
            "choices": [" " + row["sol1"], " " + row["sol2"]],
            "label": int(row["label"]),
        }


def load_winogrande(limit=None):
    """Esquema parcial: troca o '_' por cada opcao e pontua so o sufixo comum."""
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        idx = row["sentence"].index("_")
        suffix = row["sentence"][idx + 1:]
        yield {
            "contexts": [row["sentence"][:idx] + row["option1"],
                         row["sentence"][:idx] + row["option2"]],
            "choices": [suffix, suffix],
            "label": int(row["answer"]) - 1,
        }


TASKS = {
    "hellaswag": load_hellaswag,
    "arc_easy": lambda limit=None: load_arc("ARC-Easy", limit),
    "arc_challenge": lambda limit=None: load_arc("ARC-Challenge", limit),
    "piqa": load_piqa,
    "winogrande": load_winogrande,
}


@torch.no_grad()
def score_example(model, example, device, max_seq_len) -> tuple[int, int]:
    """Devolve (indice da maior log-verossimilhanca, idem normalizada)."""
    contexts = example.get("contexts") or [example["context"]] * len(example["choices"])

    seqs, starts = [], []
    for ctx, choice in zip(contexts, example["choices"]):
        ctx_ids = ENC.encode_ordinary(ctx)
        cont_ids = ENC.encode_ordinary(choice)
        ids = ctx_ids + cont_ids
        if len(ids) > max_seq_len:              # trunca pela esquerda
            cut = len(ids) - max_seq_len
            ids = ids[cut:]
            ctx_len = max(1, len(ctx_ids) - cut)
        else:
            ctx_len = len(ctx_ids)
        seqs.append(ids)
        starts.append(ctx_len)

    lengths = [len(s) for s in seqs]
    maxlen = max(lengths)
    x = torch.zeros(len(seqs), maxlen, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _, _ = model(x, full_logits=True)
    logprobs = F.log_softmax(logits.float(), dim=-1)

    # o token da posicao t e previsto pelos logits da posicao t-1
    targets = x[:, 1:]
    token_lp = logprobs[:, :-1, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    pos = torch.arange(1, maxlen, device=device).unsqueeze(0)       # posicao alvo t
    starts_t = torch.tensor(starts, device=device).unsqueeze(1)
    lens_t = torch.tensor(lengths, device=device).unsqueeze(1)
    keep = (pos >= starts_t) & (pos < lens_t)                       # so a continuacao

    scores = (token_lp * keep).sum(dim=-1)
    char_len = torch.tensor([max(1, len(c)) for c in example["choices"]],
                            device=device, dtype=scores.dtype)
    norm_scores = scores / char_len

    return int(scores.argmax().item()), int(norm_scores.argmax().item())


def run_task(model, name, device, max_seq_len, limit=None):
    loader = TASKS[name]
    correct = correct_norm = total = n_choices = 0
    for example in loader(limit):
        pred, pred_norm = score_example(model, example, device, max_seq_len)
        correct += int(pred == example["label"])
        correct_norm += int(pred_norm == example["label"])
        n_choices = len(example["choices"])
        total += 1
        if total % 500 == 0:
            print(f"    {name}: {total} exemplos | acc {correct/total:.4f} "
                  f"| acc_norm {correct_norm/total:.4f}")
    return {
        "n": total,
        "acc": correct / max(1, total),
        "acc_norm": correct_norm / max(1, total),
        "random_baseline": 1.0 / n_choices if n_choices else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sft_final.pt")
    ap.add_argument("--val-path", default="data/pretrain/val.npy")
    ap.add_argument("--tasks", default="hellaswag,arc_easy",
                    help="lista separada por virgula; vazio pula os benchmarks")
    ap.add_argument("--limit", type=int, default=None,
                    help="limita exemplos por tarefa (execucao rapida)")
    ap.add_argument("--ppl-batches", type=int, default=None,
                    help="batches do split de validacao; None = tudo")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default="logs/eval_results.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "este script precisa de GPU CUDA"
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    model, _ = load_model_from_checkpoint(args.ckpt, device)
    model.eval()
    seq_len = model.cfg.max_seq_len
    results = {"checkpoint": args.ckpt, "params": model.num_params()}

    print(f"checkpoint: {args.ckpt} | {model.num_params()/1e6:.1f}M parametros\n")

    if os.path.exists(args.val_path):
        loss, ppl = perplexity(model, args.val_path, seq_len, args.batch_size,
                               args.ppl_batches, device)
        results["val_loss"] = round(loss, 5)
        results["perplexity"] = round(ppl, 3)
        print(f"Perplexidade ({args.val_path}): {ppl:.3f}  (loss {loss:.4f})\n")
    else:
        print(f"[aviso] {args.val_path} nao existe; perplexidade pulada\n")

    for name in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        if name not in TASKS:
            print(f"[aviso] tarefa desconhecida: {name}")
            continue
        print(f"Rodando {name}...")
        try:
            res = run_task(model, name, device, seq_len, args.limit)
        except Exception as exc:
            print(f"  [erro] {name}: {exc}")
            continue
        results[name] = res
        print(f"  {name}: acc {res['acc']:.4f} | acc_norm {res['acc_norm']:.4f} "
              f"| n={res['n']} | aleatorio {res['random_baseline']:.3f}\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"resultados salvos em {args.out}")


if __name__ == "__main__":
    main()
