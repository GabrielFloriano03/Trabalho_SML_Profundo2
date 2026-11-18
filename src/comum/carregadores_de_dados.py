"""Leitura dos dados tokenizados.

Os tres estagios usam o mesmo formato: um fluxo plano de ids uint16 em arquivos
.npy. O SFT carrega, alem dos ids, uma mascara uint8 que marca quais tokens
pertencem a resposta do assistente (os unicos que entram na loss).

A leitura e sequencial e retomavel: o estado (shard atual + posicao) entra no
checkpoint, entao um treino interrompido continua exatamente de onde parou.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch


class ShardLoader:
    """Percorre shards .npy em sequencia, entregando batches (B, T) contiguos."""

    def __init__(self, data_dir: str, pattern: str, batch_size: int, seq_len: int,
                 device: str = "cuda", mask_pattern: str | None = None,
                 loop: bool = True):
        self.files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not self.files:
            raise FileNotFoundError(
                f"nenhum arquivo casa com {pattern} em {data_dir}. "
                f"Rode o script de preparacao de dados antes."
            )
        self.mask_files = (sorted(glob.glob(os.path.join(data_dir, mask_pattern)))
                           if mask_pattern else None)
        if self.mask_files is not None and len(self.mask_files) != len(self.files):
            raise ValueError("numero de shards de ids e de mascaras nao bate")

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.loop = loop
        self.epochs = 0

        self.shard_idx = 0
        self.pos = 0
        self._load_shard(0)

    # ---- estado (para resume) -------------------------------------------------
    def state_dict(self) -> dict:
        return {"shard_idx": self.shard_idx, "pos": self.pos, "epochs": self.epochs}

    def load_state_dict(self, state: dict) -> None:
        self.epochs = state.get("epochs", 0)
        self._load_shard(state["shard_idx"])
        self.pos = state["pos"]

    # ---- interno --------------------------------------------------------------
    def _load_shard(self, idx: int) -> None:
        self.shard_idx = idx
        self.tokens = np.load(self.files[idx], mmap_mode="r")
        self.mask = (np.load(self.mask_files[idx], mmap_mode="r")
                     if self.mask_files else None)
        self.pos = 0

    @property
    def total_tokens(self) -> int:
        return sum(len(np.load(f, mmap_mode="r")) for f in self.files)

    def next_batch(self):
        """Devolve (x, y) em (B, T). y traz -100 onde a loss deve ser ignorada."""
        need = self.batch_size * self.seq_len + 1

        if self.pos + need > len(self.tokens):
            nxt = self.shard_idx + 1
            if nxt >= len(self.files):
                if not self.loop:
                    raise StopIteration
                nxt = 0
                self.epochs += 1
            self._load_shard(nxt)

        chunk = np.asarray(self.tokens[self.pos:self.pos + need], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1]).view(self.batch_size, self.seq_len)
        y = torch.from_numpy(chunk[1:]).view(self.batch_size, self.seq_len)

        if self.mask is not None:
            # a mascara acompanha os targets, ou seja, as posicoes de chunk[1:]
            m = np.asarray(self.mask[self.pos + 1:self.pos + need], dtype=np.int64)
            m = torch.from_numpy(m).view(self.batch_size, self.seq_len)
            y = y.masked_fill(m == 0, -100)

        self.pos += self.batch_size * self.seq_len

        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


class ValLoader:
    """Iterador deterministico sobre o split de validacao, para perplexidade."""

    def __init__(self, path: str, batch_size: int, seq_len: int, device: str = "cuda",
                 mask_path: str | None = None):
        self.tokens = np.load(path, mmap_mode="r")
        self.mask = np.load(mask_path, mmap_mode="r") if mask_path else None
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.step = self.batch_size * self.seq_len
        self.n_batches = (len(self.tokens) - 1) // self.step

    def batches(self, max_batches: int | None = None):
        n = self.n_batches if max_batches is None else min(max_batches, self.n_batches)
        for i in range(n):
            start = i * self.step
            chunk = np.asarray(self.tokens[start:start + self.step + 1], dtype=np.int64)
            x = torch.from_numpy(chunk[:-1]).view(self.batch_size, self.seq_len)
            y = torch.from_numpy(chunk[1:]).view(self.batch_size, self.seq_len)
            if self.mask is not None:
                m = np.asarray(self.mask[start + 1:start + self.step + 1], dtype=np.int64)
                m = torch.from_numpy(m).view(self.batch_size, self.seq_len)
                y = y.masked_fill(m == 0, -100)
            yield x.to(self.device), y.to(self.device)
