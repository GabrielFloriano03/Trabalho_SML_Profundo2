"""Geracao de texto a partir de um checkpoint.

Dois modos, que cobrem o item 5 da entrega:
  --mode raw   : completamento livre de prompt (modelo pre-treinado)
  --mode chat  : pergunta -> resposta no template de chat (modelo pos-SFT)

Uso:
    python src/generate.py --ckpt checkpoints/pretrain_final.pt --mode raw
    python src/generate.py --ckpt checkpoints/sft_final.pt --mode chat
    python src/generate.py --ckpt ... --prompt "The capital of France is"
"""

from __future__ import annotations

import argparse
import json
import os

import tiktoken
import torch

from chat_format import build_prompt
from train_utils import load_model_from_checkpoint

ENC = tiktoken.get_encoding("gpt2")

PROMPTS_RAW = [
    "The capital of France is",
    "Photosynthesis is the process by which",
    "In mathematics, a prime number is",
    "The water cycle describes how",
    "Isaac Newton is best known for",
]

PROMPTS_CHAT = [
    "What is the capital of France?",
    "Explain photosynthesis in simple terms.",
    "Write a short poem about the ocean.",
    "If I have 3 apples and buy 7 more, how many do I have?",
    "Give me three tips to study more effectively.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sft_final.pt")
    ap.add_argument("--mode", choices=["raw", "chat"], default="chat")
    ap.add_argument("--prompt", default=None, help="um prompt unico; sem isso usa a lista padrao")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=None, help="salva as amostras em JSON")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "este script precisa de GPU CUDA"
    device = "cuda"
    torch.manual_seed(args.seed)

    model, _ = load_model_from_checkpoint(args.ckpt, device)
    model.eval()
    print(f"checkpoint: {args.ckpt} | {model.num_params()/1e6:.1f}M parametros")
    print(f"modo: {args.mode} | T={args.temperature} top_k={args.top_k}\n")

    prompts = [args.prompt] if args.prompt else (
        PROMPTS_CHAT if args.mode == "chat" else PROMPTS_RAW)

    samples = []
    for prompt in prompts:
        text = (build_prompt([{"role": "user", "content": prompt}])
                if args.mode == "chat" else prompt)
        ids = torch.tensor([[ENC.eot_token] + ENC.encode_ordinary(text)],
                           dtype=torch.long, device=device)
        ids = ids.repeat(args.num_samples, 1)

        out = model.generate(ids, args.max_new_tokens, args.temperature,
                             args.top_k, eos_id=ENC.eot_token)

        print("=" * 72)
        print(f"PROMPT: {prompt}")
        for row in out:
            gen = row[ids.size(1):].tolist()
            if ENC.eot_token in gen:
                gen = gen[:gen.index(ENC.eot_token)]
            completion = ENC.decode(gen)
            print("-" * 72)
            print(completion.strip())
            samples.append({"mode": args.mode, "prompt": prompt,
                            "completion": completion.strip()})
        print()

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": args.ckpt, "samples": samples}, f,
                      indent=2, ensure_ascii=False)
        print(f"amostras salvas em {args.out}")


if __name__ == "__main__":
    main()
