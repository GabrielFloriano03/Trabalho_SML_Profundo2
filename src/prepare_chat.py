"""Prepara os dados conversacionais do mid-training e do SFT.

Converte SmolTalk (+ GSM8K, opcional) para o template de chat, aplica sequence
packing em um fluxo continuo de tokens e grava ids + mascara de loss.

Saida em --out:
    train_0000.npy / train_mask_0000.npy   (uint16 / uint8)
    val.npy        / val_mask.npy

Uso:
    python src/prepare_chat.py --out data/midtrain --max-tokens 150e6
    python src/prepare_chat.py --out data/sft      --max-tokens 40e6
"""

from __future__ import annotations

import argparse
import os
import re
import time

import numpy as np
from datasets import load_dataset

from chat_format import encode_conversation


def human(n: float) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}T"


def iter_smoltalk(config: str):
    ds = load_dataset("HuggingFaceTB/smoltalk", config, split="train", streaming=True)
    for row in ds:
        msgs = row.get("messages")
        if msgs:
            yield msgs


# anotacoes de calculadora do GSM8K, ex. "7 * 7 = <<7*7=49>>49"
CALC_ANNOT = re.compile(r"<<[^>]*>>")


def iter_gsm8k():
    """GSM8K como conversa; a resposta mantem o raciocinio passo a passo.

    O texto original do dataset precisa de duas limpezas:

    - `<<7*7=49>>` sao anotacoes de calculadora usadas na construcao do GSM8K,
      nao parte da resposta. Mantidas, o modelo aprende a emitir uma sintaxe
      que nao tem como executar, e produz contas erradas dentro dela.
    - o marcador `####` vira "The answer is:". O corpus de pre-treino e de
      mid-training e inteiramente em ingles; usar uma expressao em outro idioma
      aqui faz o modelo misturar idiomas nas respostas.
    """
    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    for row in ds:
        answer = CALC_ANNOT.sub("", row["answer"]).replace("####", "The answer is:")
        yield [{"role": "user", "content": row["question"]},
               {"role": "assistant", "content": answer}]


def interleave(sources: list, weights: list[int]):
    """Alterna entre as fontes na proporcao dada, ate todas se esgotarem."""
    iters = [iter(s) for s in sources]
    alive = [True] * len(iters)
    while any(alive):
        for i, it in enumerate(iters):
            if not alive[i]:
                continue
            for _ in range(weights[i]):
                try:
                    yield next(it)
                except StopIteration:
                    alive[i] = False
                    break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/midtrain")
    ap.add_argument("--smoltalk-config", default="all",
                    help="um config ou varios separados por virgula, ex. "
                         "smol-magpie-ultra,everyday-conversations")
    ap.add_argument("--with-gsm8k", action="store_true", default=True)
    ap.add_argument("--no-gsm8k", dest="with_gsm8k", action="store_false")
    ap.add_argument("--gsm8k-weight", type=int, default=1,
                    help="conversas de GSM8K a cada --smoltalk-weight de SmolTalk")
    ap.add_argument("--smoltalk-weight", type=int, default=12)
    ap.add_argument("--max-tokens", type=float, default=150e6)
    ap.add_argument("--val-tokens", type=float, default=2e6)
    ap.add_argument("--shard-tokens", type=float, default=1e8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    target = int(args.max_tokens)
    val_target = int(args.val_tokens)
    shard_size = int(args.shard_tokens)

    configs = [c.strip() for c in args.smoltalk_config.split(",") if c.strip()]
    sources = [iter_smoltalk(c) for c in configs]
    weights = [args.smoltalk_weight] * len(configs)
    if args.with_gsm8k:
        sources.append(iter_gsm8k())
        weights.append(args.gsm8k_weight)

    print(f"destino : {args.out}")
    print("fontes  : " + " + ".join(f"SmolTalk[{c}]" for c in configs)
          + (" + GSM8K" if args.with_gsm8k else ""))
    print(f"meta    : {human(target)} tokens de treino + {human(val_target)} de validacao\n")

    ids_buf = np.empty((shard_size,), dtype=np.uint16)
    mask_buf = np.empty((shard_size,), dtype=np.uint8)
    buf_n = 0
    shard_idx = 0
    written = 0
    val_ids: list[int] = []
    val_mask: list[int] = []
    val_done = False
    t0 = time.time()

    def flush():
        nonlocal buf_n, shard_idx, written
        if buf_n == 0:
            return
        np.save(os.path.join(args.out, f"train_{shard_idx:04d}.npy"), ids_buf[:buf_n])
        np.save(os.path.join(args.out, f"train_mask_{shard_idx:04d}.npy"), mask_buf[:buf_n])
        written += buf_n
        print(f"  shard {shard_idx:04d}: {human(buf_n)} tokens | total {human(written)} "
              f"| {time.time()-t0:.0f}s")
        shard_idx += 1
        buf_n = 0

    n_conv = 0
    for messages in interleave(sources, weights):
        ids, mask = encode_conversation(messages)
        n_conv += 1

        if not val_done:
            val_ids.extend(ids)
            val_mask.extend(mask)
            if len(val_ids) >= val_target:
                np.save(os.path.join(args.out, "val.npy"),
                        np.array(val_ids[:val_target], dtype=np.uint16))
                np.save(os.path.join(args.out, "val_mask.npy"),
                        np.array(val_mask[:val_target], dtype=np.uint8))
                print(f"  val: {human(val_target)} tokens")
                val_ids, val_mask, val_done = [], [], True
            continue

        arr_ids = np.array(ids, dtype=np.uint16)
        arr_mask = np.array(mask, dtype=np.uint8)
        pos = 0
        while pos < len(arr_ids):
            take = min(len(ids_buf) - buf_n, len(arr_ids) - pos)
            ids_buf[buf_n:buf_n + take] = arr_ids[pos:pos + take]
            mask_buf[buf_n:buf_n + take] = arr_mask[pos:pos + take]
            buf_n += take
            pos += take
            if buf_n == len(ids_buf):
                flush()

        if written + buf_n >= target:
            break

    flush()
    print(f"\nPronto: {human(written)} tokens em {shard_idx} shards "
          f"({n_conv:,} conversas), {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
