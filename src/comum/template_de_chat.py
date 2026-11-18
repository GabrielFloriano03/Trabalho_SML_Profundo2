"""Template de chat usado no mid-training, no SFT e na demo.

Formato (o sugerido no enunciado), com <|endoftext|> separando conversas:

    <|endoftext|>User: pergunta
    Assistant: resposta<|endoftext|>

A funcao encode_conversation devolve ids e uma mascara: 1 nos tokens que o
modelo deve aprender a gerar (a resposta do assistente e o EOT final), 0 nos
tokens de prompt. O SFT usa essa mascara para zerar a loss do prompt; o
mid-training ignora a mascara e treina em todos os tokens.
"""

from __future__ import annotations

import tiktoken

ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token

USER_PREFIX = "User: "
ASSISTANT_PREFIX = "Assistant:"


def build_prompt(messages: list[dict]) -> str:
    """Monta o texto do prompt para inferencia (termina em 'Assistant:')."""
    parts = []
    for m in messages:
        if m["role"] == "user":
            parts.append(f"{USER_PREFIX}{m['content']}\n{ASSISTANT_PREFIX}")
        elif m["role"] == "assistant":
            parts.append(f" {m['content']}\n")
    return "".join(parts)


def encode_conversation(messages: list[dict]) -> tuple[list[int], list[int]]:
    """Conversa -> (ids, mask). mask=1 somente nos tokens do assistente."""
    ids: list[int] = [EOT]
    mask: list[int] = [0]

    for m in messages:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if not content:
            continue
        if role in ("user", "human"):
            seg = ENC.encode_ordinary(f"{USER_PREFIX}{content}\n{ASSISTANT_PREFIX}")
            ids.extend(seg)
            mask.extend([0] * len(seg))
        elif role in ("assistant", "gpt"):
            seg = ENC.encode_ordinary(f" {content}\n")
            ids.extend(seg)
            mask.extend([1] * len(seg))
        elif role == "system":
            seg = ENC.encode_ordinary(f"{content}\n")
            ids.extend(seg)
            mask.extend([0] * len(seg))

    ids.append(EOT)
    mask.append(1)
    return ids, mask
