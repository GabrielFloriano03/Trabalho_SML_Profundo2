"""Etapa 2: mid-training.

Parte dos pesos do pre-treino (sem o estado do otimizador) e continua o treino
em dados conversacionais e de raciocinio, com LR menor e warmup curto.
A loss cobre TODOS os tokens -- a mascara so entra no SFT.

Uso:
    python src/midtrain.py
    python src/midtrain.py --resume
"""

from __future__ import annotations

import argparse

from finetune_core import add_common_args, run_stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/midtrain")
    ap.add_argument("--init-from", default="checkpoints/pretrain_final.pt")
    ap.add_argument("--total-tokens", type=float, default=150e6,
                    help="0 = usar --epochs sobre o dataset inteiro")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=2e-5)
    ap.add_argument("--warmup-steps", type=int, default=50)
    add_common_args(ap, "midtrain")
    args = ap.parse_args()

    run_stage(args, stage="midtrain", use_loss_mask=False)


if __name__ == "__main__":
    main()
