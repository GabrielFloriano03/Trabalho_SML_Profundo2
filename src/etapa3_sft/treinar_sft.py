"""Etapa 3: supervised fine-tuning (SFT).

Parte dos pesos do mid-training e treina com loss mask: so os tokens da
resposta do assistente contam na loss; os tokens de prompt (User: ...) entram
como -100 e sao ignorados pela cross-entropy. LR menor ainda que no
mid-training, 1 a 2 epocas, monitorando overfitting pela val loss.

Uso:
    python src/etapa3_sft/treinar_sft.py
    python src/etapa3_sft/treinar_sft.py --epochs 2 --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# os modulos compartilhados entre as etapas ficam em src/comum/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comum"))

from loop_de_finetuning import add_common_args, run_stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/sft")
    ap.add_argument("--init-from", default="checkpoints/midtrain_final.pt")
    ap.add_argument("--total-tokens", type=float, default=0,
                    help="0 = usar --epochs sobre o dataset inteiro")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--min-lr", type=float, default=2e-6)
    ap.add_argument("--warmup-steps", type=int, default=30)
    add_common_args(ap, "sft")
    args = ap.parse_args()

    run_stage(args, stage="sft", use_loss_mask=True)


if __name__ == "__main__":
    main()
