"""Gera os graficos das curvas de loss das tres etapas (item 3 da entrega).

Uso:
    python src/plot_curves.py
    python src/plot_curves.py --x tokens --out logs/curvas.png
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STAGES = [("pretrain", "Pre-treino"), ("midtrain", "Mid-training"), ("sft", "SFT")]


def read_csv(path: str):
    train_x, train_y, val_x, val_y, ppl_x, ppl_y = [], [], [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("train_loss"):
                train_x.append(row)
                train_y.append(float(row["train_loss"]))
            if row.get("val_loss"):
                val_x.append(row)
                val_y.append(float(row["val_loss"]))
            if row.get("val_ppl"):
                ppl_x.append(row)
                ppl_y.append(float(row["val_ppl"]))
    return (train_x, train_y), (val_x, val_y), (ppl_x, ppl_y)


def xs(rows, key):
    return [float(r[key]) for r in rows]


def smooth(ys, k=15):
    if len(ys) < k:
        return ys
    out, acc = [], 0.0
    for i, y in enumerate(ys):
        acc += y
        if i >= k:
            acc -= ys[i - k]
        out.append(acc / min(i + 1, k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--x", choices=["step", "tokens"], default="tokens")
    ap.add_argument("--out", default="logs/curvas_loss.png")
    args = ap.parse_args()

    present = [(s, label, os.path.join(args.logs_dir, f"{s}.csv"))
               for s, label in STAGES
               if os.path.exists(os.path.join(args.logs_dir, f"{s}.csv"))]
    if not present:
        raise SystemExit(f"nenhum CSV de treino encontrado em {args.logs_dir}/")

    fig, axes = plt.subplots(2, len(present), figsize=(5.5 * len(present), 8),
                             squeeze=False)
    scale = 1e9 if args.x == "tokens" else 1
    xlabel = "tokens (B)" if args.x == "tokens" else "step"

    for col, (stage, label, path) in enumerate(present):
        (tr_rows, tr_y), (va_rows, va_y), (pp_rows, pp_y) = read_csv(path)

        ax = axes[0][col]
        if tr_y:
            x = [v / scale for v in xs(tr_rows, args.x)]
            ax.plot(x, tr_y, alpha=0.25, lw=0.8, color="tab:blue")
            ax.plot(x, smooth(tr_y), lw=1.6, color="tab:blue", label="treino")
        if va_y:
            ax.plot([v / scale for v in xs(va_rows, args.x)], va_y, "o-",
                    ms=3, lw=1.4, color="tab:red", label="validacao")
        ax.set_title(f"{label} - loss")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("cross-entropy")
        ax.grid(alpha=0.3)
        ax.legend()

        ax = axes[1][col]
        if pp_y:
            ax.plot([v / scale for v in xs(pp_rows, args.x)], pp_y, "o-",
                    ms=3, lw=1.4, color="tab:green")
            ax.set_yscale("log")
        ax.set_title(f"{label} - perplexidade (validacao)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PPL")
        ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"grafico salvo em {args.out}")


if __name__ == "__main__":
    main()
