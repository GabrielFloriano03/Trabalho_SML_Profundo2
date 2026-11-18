"""Baixa (em streaming) e tokeniza o corpus de pre-treino em shards uint16.

Corpus: FineWeb-Edu, config sample-10BT (HuggingFaceFW/fineweb-edu).
O streaming evita guardar o parquet bruto em disco: so os shards tokenizados
ficam salvos (~2 bytes por token).

Os primeiros --val-tokens tokens viram o split de validacao (val.npy) e nunca
entram no treino -- e sobre ele que a perplexidade e reportada.

Uso:
    python src/etapa0_preparacao_dos_dados/preparar_corpus_de_pretreino.py --target-tokens 1.1e9

Estrategia de shards inspirada em build-nanoGPT (Karpathy, MIT).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import tiktoken
from datasets import load_dataset

ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token  # 50256, separa documentos


def tokenize_doc(doc) -> np.ndarray:
    """Documento -> array uint16 com <|endoftext|> na frente."""
    ids = [EOT]
    ids.extend(ENC.encode_ordinary(doc["text"]))
    arr = np.array(ids, dtype=np.uint32)
    assert (arr < 2 ** 16).all(), "token fora do alcance do uint16"
    return arr.astype(np.uint16)


def human(n: float) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}T"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pretrain")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--name", default="sample-10BT")
    ap.add_argument("--target-tokens", type=float, default=1.1e9)
    ap.add_argument("--val-tokens", type=float, default=5e6)
    ap.add_argument("--shard-tokens", type=float, default=1e8)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    args = ap.parse_args()

    target = int(args.target_tokens)
    val_tokens = int(args.val_tokens)
    shard_size = int(args.shard_tokens)

    os.makedirs(args.out, exist_ok=True)
    print(f"corpus   : {args.dataset} [{args.name}]")
    print(f"destino  : {args.out}")
    print(f"meta     : {human(target)} tokens de treino + {human(val_tokens)} de validacao")
    print(f"workers  : {args.workers}\n")

    ds = load_dataset(args.dataset, name=args.name, split="train", streaming=True)

    buf = np.empty((shard_size,), dtype=np.uint16)
    buf_n = 0
    shard_idx = 0
    written_train = 0
    val_done = os.path.exists(os.path.join(args.out, "val.npy"))
    val_buf: list[np.ndarray] = []
    val_n = 0
    t0 = time.time()

    def flush_shard():
        nonlocal buf_n, shard_idx, written_train
        if buf_n == 0:
            return
        path = os.path.join(args.out, f"train_{shard_idx:04d}.npy")
        np.save(path, buf[:buf_n])
        written_train += buf_n
        print(f"  shard {shard_idx:04d}: {human(buf_n)} tokens -> {path} "
              f"| total {human(written_train)} | {time.time()-t0:.0f}s")
        shard_idx += 1
        buf_n = 0

    with mp.Pool(args.workers) as pool:
        for tokens in pool.imap(tokenize_doc, ds, chunksize=16):
            # 1) preencher primeiro o split de validacao
            if not val_done:
                val_buf.append(tokens)
                val_n += len(tokens)
                if val_n >= val_tokens:
                    val = np.concatenate(val_buf)[:val_tokens]
                    np.save(os.path.join(args.out, "val.npy"), val)
                    print(f"  val.npy: {human(len(val))} tokens")
                    val_buf, val_done = [], True
                continue

            # 2) depois alimentar os shards de treino
            pos = 0
            while pos < len(tokens):
                space = len(buf) - buf_n
                take = min(space, len(tokens) - pos)
                buf[buf_n:buf_n + take] = tokens[pos:pos + take]
                buf_n += take
                pos += take
                if buf_n == len(buf):
                    flush_shard()

            if written_train + buf_n >= target:
                break

        flush_shard()

    print(f"\nPronto: {human(written_train)} tokens de treino em {shard_idx} shards, "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
