"""Compara dois checkpoints de SFT quanto ao vazamento de artefatos do GSM8K.

Mede duas coisas nas respostas geradas:
  - anotacoes de calculadora do GSM8K, padrao <<numeros e operadores>>
  - o marcador em portugues "A resposta e", introduzido por engano na primeira
    versao da preparacao de dados

Usa prompts aritmeticos, que sao os que ativam o comportamento aprendido do
GSM8K; varias amostras por prompt porque a geracao e estocastica (T=0.8) e uma
unica amostra limpa nao prova ausencia do vazamento.

O padrao de busca e restrito a numeros e operadores de proposito: conversas de
codigo do SmolTalk contem `std::cout << ... >>` legitimamente, e um padrao
largo casaria com elas.

Uso (a partir da raiz do projeto):
    python src/check_leak.py
"""

from __future__ import annotations

import json
import os
import re
import sys

import tiktoken
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat_format import build_prompt  # noqa: E402
from train_utils import load_model_from_checkpoint  # noqa: E402

ENC = tiktoken.get_encoding("gpt2")
CALC = re.compile(r"<<[\d\s()+*/.,=-]+>>")

PROMPTS = [
    "If I have 3 apples and buy 7 more, how many do I have?",
    "Natalia sold 20 clips in April and half as many in May. How many in total?",
    "A book costs $12. How much do 5 books cost?",
    "Tom runs 3 miles a day for 7 days. How far did he run?",
    "There are 24 students. If 1/3 leave, how many remain?",
    "What is 15 plus 27?",
    "Sarah has 5 boxes with 8 pencils each. How many pencils?",
    "A shirt costs $20 with a 25% discount. What is the final price?",
]
N_AMOSTRAS = 4


def avaliar(ckpt: str, device: str = "cuda") -> dict:
    model, _ = load_model_from_checkpoint(ckpt, device)
    model.eval()
    torch.manual_seed(1337)

    calc_hits, pt_hits, textos = 0, 0, []
    for prompt in PROMPTS:
        texto = build_prompt([{"role": "user", "content": prompt}])
        ids = torch.tensor([[ENC.eot_token] + ENC.encode_ordinary(texto)],
                           dtype=torch.long, device=device).repeat(N_AMOSTRAS, 1)
        out = model.generate(ids, 140, 0.8, 50, eos_id=ENC.eot_token)
        for row in out:
            gen = row[ids.size(1):].tolist()
            if ENC.eot_token in gen:
                gen = gen[:gen.index(ENC.eot_token)]
            resp = ENC.decode(gen).strip()
            calc_hits += len(CALC.findall(resp))
            pt_hits += resp.count("A resposta e")
            textos.append({"prompt": prompt, "resposta": resp})

    return {"checkpoint": ckpt, "amostras": len(PROMPTS) * N_AMOSTRAS,
            "anotacoes_calculadora": calc_hits,
            "marcador_portugues": pt_hits, "textos": textos}


def main():
    alvos = [("v1 (dados sujos)", os.path.join("checkpoints", "sft_final_v1.pt")),
             ("v2 (dados limpos)", os.path.join("checkpoints", "sft_final.pt"))]

    resultados = {}
    for rotulo, ckpt in alvos:
        if not os.path.exists(ckpt):
            print(f"[{rotulo}] {ckpt} nao encontrado, pulando")
            continue
        r = avaliar(ckpt)
        resultados[rotulo] = r
        print(f"[{rotulo}] {r['amostras']} amostras | "
              f"<<calculadora>>={r['anotacoes_calculadora']} | "
              f"'A resposta e'={r['marcador_portugues']}")

    os.makedirs("logs", exist_ok=True)
    destino = os.path.join("logs", "teste_vazamento.json")
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"\ndetalhes em {destino}")


if __name__ == "__main__":
    main()
