"""Arquitetura do SLM: transformer decoder-only no estilo LLaMA.

Escolhas de design (justificativa no README):
  - RoPE                  : extrapola melhor e nao gasta parametros com embedding posicional
  - GQA                   : menos parametros/memoria em K,V e KV-cache menor na geracao
  - SwiGLU                : melhor qualidade por FLOP que GELU/MLP classico
  - RMSNorm (pre-norm)    : mais barato que LayerNorm e estabiliza o treino profundo
  - Flash Attention (SDPA): kernel fundido do PyTorch, essencial com 6 GB de VRAM
  - Weight tying          : embedding de entrada e saida compartilhadas

Referencia de estrutura: nanoGPT (Andrej Karpathy, MIT) e o paper do LLaMA.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SLMConfig:
    """Hiperparametros da arquitetura.

    O default fecha em ~55M parametros treinaveis, dentro da faixa 50M-200M
    pedida no enunciado.
    """

    vocab_size: int = 50304        # 50257 do BPE do GPT-2, arredondado p/ multiplo de 64
    real_vocab_size: int = 50257   # ids validos do tokenizer (o resto e padding)
    n_layer: int = 10
    n_head: int = 8
    n_kv_head: int = 4             # GQA: 2 query heads por kv head
    d_model: int = 512
    ffn_hidden: int = 1408         # ~8/3 * d_model, multiplo de 64
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # a normalizacao e feita em fp32 para nao perder precisao sob autocast
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """Pre-computa cos/sin do RoPE para todas as posicoes ate seq_len."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)              # (T, head_dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotaciona x (B, n_head, T, head_dim) na variante 'metade-metade'."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class Attention(nn.Module):
    """Grouped-Query Attention causal com Flash Attention via SDPA."""

    def __init__(self, cfg: SLMConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        new_cache = None
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            if k_prev is not None:
                k = torch.cat((k_prev, k), dim=2)
                v = torch.cat((v_prev, v), dim=2)
            new_cache = (k, v)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # no decode incremental (T == 1) a query ve todo o cache, entao nao ha mascara
        is_causal = T > 1
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y), new_cache


class SwiGLU(nn.Module):
    """Feed-forward SwiGLU (Shazeer, 2020)."""

    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Bloco pre-norm: x + Attn(norm(x)); depois x + FFN(norm(x))."""

    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        attn_out, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class SLM(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta,
                                    device="cpu", dtype=torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # init escalado nas projecoes residuais (GPT-2 / nanoGPT)
        scale = 1.0 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx, targets=None, kv_caches=None, start_pos: int = 0,
                full_logits: bool = False):
        """idx: (B, T) de ids; targets: (B, T) com -100 nas posicoes mascaradas.

        full_logits=True devolve os logits de todas as posicoes mesmo sem
        targets (usado na avaliacao por log-likelihood).
        """
        B, T = idx.shape
        assert start_pos + T <= self.cfg.max_seq_len, "sequencia maior que max_seq_len"

        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]

        x = self.drop(self.tok_emb(idx))

        new_caches = [] if kv_caches is not None else None
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, cos, sin, cache_i)
            if new_caches is not None:
                new_caches.append(new_cache)

        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=-100,
            )
            return logits, loss, new_caches

        if full_logits:
            return self.lm_head(x), None, new_caches

        # em inferencia so o ultimo passo importa -> economiza memoria
        logits = self.lm_head(x[:, -1:, :])
        return logits, None, new_caches

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """AdamW sem weight decay em tensores 1-D (normas)."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, eps=1e-8,
                                 fused=(device_type == "cuda"))

    @torch.no_grad()
    def generate_iter(self, idx, max_new_tokens, temperature=0.8, top_k=50,
                      eos_id=None, use_cache=True):
        """Geracao autorregressiva com KV-cache, um token por vez.

        Devolve cada token amostrado como um tensor (B, 1), na ordem em que sai
        do modelo. O EOS tambem e devolvido antes de encerrar, para quem consome
        poder decidir se corta ou mostra.

        E aqui que vive a amostragem; `generate` apenas acumula o resultado.
        A demo em modo chat consome este gerador para exibir a resposta
        aparecendo token a token.
        """
        was_training = self.training
        self.eval()
        cfg = self.cfg

        try:
            if idx.size(1) >= cfg.max_seq_len:
                idx = idx[:, -(cfg.max_seq_len - 1):]

            kv_caches = [(None, None) for _ in range(cfg.n_layer)] if use_cache else None
            logits, _, kv_caches = self(idx, kv_caches=kv_caches, start_pos=0)
            pos = idx.size(1)

            for _ in range(max_new_tokens):
                step = logits[:, -1, :].float()
                step[:, cfg.real_vocab_size:] = float("-inf")  # nunca amostrar padding

                if temperature <= 0.0:
                    next_id = step.argmax(dim=-1, keepdim=True)
                else:
                    step = step / temperature
                    if top_k is not None and top_k > 0:
                        k = min(top_k, step.size(-1))
                        thresh = torch.topk(step, k, dim=-1).values[:, -1:]
                        step = step.masked_fill(step < thresh, float("-inf"))
                    probs = F.softmax(step, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)

                idx = torch.cat((idx, next_id), dim=1)
                yield next_id

                if eos_id is not None and (next_id == eos_id).all():
                    break
                if pos + 1 >= cfg.max_seq_len:
                    break

                if use_cache:
                    logits, _, kv_caches = self(next_id, kv_caches=kv_caches,
                                                start_pos=pos)
                    pos += 1
                else:
                    window = idx[:, -cfg.max_seq_len:]
                    logits, _, _ = self(window)
                    pos = window.size(1)
        finally:
            # tambem roda se quem consome abandonar o gerador no meio
            if was_training:
                self.train()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50,
                 eos_id=None, use_cache=True):
        """Geracao autorregressiva. Devolve o prompt seguido da resposta."""
        # a mesma janela que generate_iter aplica, para que o prompt devolvido
        # aqui seja exatamente aquele em que o modelo condicionou
        if idx.size(1) >= self.cfg.max_seq_len:
            idx = idx[:, -(self.cfg.max_seq_len - 1):]

        partes = [idx]
        partes.extend(self.generate_iter(idx, max_new_tokens, temperature, top_k,
                                         eos_id=eos_id, use_cache=use_cache))
        return torch.cat(partes, dim=1)
