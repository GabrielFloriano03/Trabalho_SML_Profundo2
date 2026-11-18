# SLM — Small Language Model treinado do zero

Projeto 2 de **Aprendizado Profundo 2** (PUCRS). Pipeline completo de um LLM em
escala reduzida: **pré-treino → mid-training → SFT → avaliação & demo**, tudo
treinado em uma única GPU de notebook.

| | |
|---|---|
| Repositório | https://github.com/GabrielFloriano03/Trabalho_SML_Profundo2 |
| Checkpoints | https://huggingface.co/Floriano03/Trabalho_SML_Profundo2 |
| Slides | [`Apresentacao_SLM_Projeto2.pdf`](Apresentacao_SLM_Projeto2.pdf) |

| | |
|---|---|
| Parâmetros treináveis | **55.26 M** (29.50 M não-embedding) |
| Arquitetura | decoder-only estilo LLaMA: RoPE · GQA · SwiGLU · RMSNorm · Flash Attention |
| Tokenizer | BPE do GPT-2 (`tiktoken`), vocab 50257 (padding até 50304) |
| Contexto | 1024 tokens |
| Corpus de pré-treino | FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, `sample-10BT`) |
| Orçamento de tokens | **1.1 B** (20 × parâmetros — Chinchilla ótimo) |
| Hardware | 1× RTX 3050 Laptop 6 GB · i7-13650HX · 16 GB RAM · Windows 11 |
| Throughput medido | 13 564 tokens/s (bf16, micro-batch 4, pico 4.52 GB) |

---

## 1. Arquitetura e justificativa das decisões

O ponto de partida é um transformer decoder-only estilo GPT. Cada componente foi
trocado por uma variante moderna, com o motivo abaixo:

| Componente | Escolha | Por quê |
|---|---|---|
| Positional encoding | **RoPE** (θ = 10000) | codifica posição relativa direto na atenção, não gasta parâmetros com uma tabela de embeddings e extrapola melhor que o posicional aprendido do GPT-2 |
| Atenção | **GQA** (8 query heads / 4 KV heads) | corta metade dos parâmetros de K e V e reduz o KV-cache pela metade na geração — com 6 GB de VRAM isso é o que viabiliza a demo com contexto cheio |
| Kernel de atenção | **Flash Attention** via `F.scaled_dot_product_attention` | kernel fundido, sem materializar a matriz T×T de atenção; sem ele o pico de memória inviabilizaria micro-batch 4 |
| Feed-forward | **SwiGLU** (hidden 1408 ≈ 8/3 · d_model) | melhor qualidade por FLOP que GELU; o fator 8/3 compensa a terceira matriz do gate e mantém a contagem de parâmetros equivalente a um MLP 4× |
| Normalização | **RMSNorm**, pre-norm | ~15% mais barata que LayerNorm (sem centragem) e o pre-norm mantém o caminho residual limpo, estabilizando o treino sem precisar de warmup longo |
| Otimizador | **AdamW fused** (β = 0.9/0.95, wd = 0.1) | `fused=True` colapsa os kernels do update, ganho real num modelo pequeno onde o overhead por step pesa; weight decay só em tensores 2-D (normas ficam de fora) |
| Schedule | **cosseno** com warmup de 200 steps | padrão bem estabelecido; `--schedule wsd` também está implementado |
| Precisão | **bf16** com autocast | sm_86 tem suporte nativo; bf16 dispensa o loss scaling do fp16 |
| Embeddings | **weight tying** (entrada = saída) | economiza 25.7 M parâmetros, que vão para as camadas |

### Contagem de parâmetros

```
embedding (tied)      50304 × 512                        = 25 755 648
por camada:
  atenção GQA         512×512 + 512×256 + 512×256 + 512×512 =    786 432
  SwiGLU              3 × 512 × 1408                       =  2 162 688
  2 × RMSNorm                                              =      1 024
                                                    total/camada = 2 950 144
10 camadas                                                 = 29 501 440
norm final                                                 =        512
                                                      TOTAL = 55 257 600
```

Dentro da faixa de 50 M–200 M pedida no enunciado.

## 2. Scaling laws — como o número de tokens foi determinado

As *Chinchilla Scaling Laws* (Hoffmann et al., 2022) indicam ≈ **20 tokens por
parâmetro** como ponto compute-ótimo:

```
55.26 M parâmetros × 20 = 1.105 B tokens
```

Arredondamos para **1.1 B tokens**, que é exatamente o ponto ótimo — não um
compromisso. O custo foi medido antes de iniciar, e não estimado:

```
13 564 tokens/s × 3600 = 48.8 M tokens/h
1.1 B / 48.8 M        ≈ 22.6 h de pré-treino
```

O micro-batch foi escolhido por medição
(`python src/ferramentas/medir_throughput_da_gpu.py`):

| micro-batch | tokens/s | pico VRAM |
|---|---|---|
| 2 | 12 223 | 2.65 GB |
| 3 | 13 226 | 3.58 GB |
| **4** | **13 564** | **4.52 GB** |
| 5 | 13 932 | 5.40 GB |
| 6 | 9 113 | 6.32 GB |
| 8 | 3 808 | 8.14 GB |

A partir de 6 o pico ultrapassa a VRAM física e o driver passa a paginar para a
RAM do sistema, derrubando o throughput em 3–4×. O micro-batch 5 é 2.6% mais
rápido que o 4, mas deixa só 0.5 GB de folga para o desktop — ficamos no **4**,
com gradient accumulation de 64 para atingir o batch global de 2¹⁸ = 262 144
tokens por passo de otimização.

## 3. Estrutura do repositório

Uma pasta por etapa do projeto, na ordem em que são executadas. Os scripts são
sempre rodados **a partir da raiz do repositório**, porque todos os caminhos
(`data/`, `checkpoints/`, `logs/`) são relativos a ela.

```
src/
│
├── etapa0_preparacao_dos_dados/
│     preparar_corpus_de_pretreino.py      FineWeb-Edu → shards uint16 + split de validação
│     preparar_conversas_de_chat.py        SmolTalk + GSM8K → shards com máscara de loss
│
├── etapa1_pretreino/
│     treinar_pretreino.py                 1,10 B tokens de texto geral, do zero
│
├── etapa2_midtraining/
│     treinar_midtraining.py               150 M tokens: aprende o formato de diálogo
│
├── etapa3_sft/
│     treinar_sft.py                       90 M tokens com loss mask: refina a resposta
│
├── etapa4_avaliacao/
│     avaliar_perplexidade_e_benchmarks.py perplexidade + 5 benchmarks por log-verossimilhança
│     gerar_amostras_de_texto.py           completamento (pré-treino) e chat (pós-SFT)
│     verificar_vazamento_do_gsm8k.py      procura artefatos do GSM8K na saída do modelo
│
├── etapa5_demo/
│     portal_de_chat.py                    portal Streamlit para a demonstração ao vivo
│
├── comum/                                 compartilhado entre as etapas
│     arquitetura_modelo.py                RoPE, GQA, SwiGLU, RMSNorm, KV-cache, geração
│     template_de_chat.py                  template de chat e máscara de loss
│     carregadores_de_dados.py             leitura dos shards, retomável
│     utilidades_de_treino.py              schedule de LR, checkpoints, logging, avaliação
│     loop_de_finetuning.py                loop compartilhado por mid-training e SFT
│
└── ferramentas/                           apoio, fora do pipeline
      medir_throughput_da_gpu.py           escolhe o micro-batch por medição
      plotar_curvas_de_loss.py             gráfico das curvas das três etapas
      gerar_slides_da_apresentacao.py      slides .pptx a partir de logs/

.streamlit/             tema do portal de chat
logs/                   CSVs das curvas, gráficos e resultados de avaliação
checkpoints/            pesos treinados (fora do git — ver seção 6)
data/                   shards tokenizados (fora do git — gerados na etapa 0)
```

## 4. Como reproduzir

### Ambiente

```bash
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verificação do pipeline antes de qualquer treino longo:

```bash
.venv\Scripts\python.exe src\ferramentas\medir_throughput_da_gpu.py --smoke
```

O smoke test confirma que o modelo consegue overfittar um mini-batch
(loss 10.91 → 0.07). A loss inicial bate com ln(50304) = 10.83, o que valida a
inicialização.

### Etapa 0 — dados

```bash
.venv\Scripts\python.exe src\etapa0_preparacao_dos_dados\preparar_corpus_de_pretreino.py --out data\pretrain --target-tokens 1.15e9

.venv\Scripts\python.exe src\etapa0_preparacao_dos_dados\preparar_conversas_de_chat.py --out data\midtrain ^
    --smoltalk-config all --max-tokens 160e6 --val-tokens 2e6

.venv\Scripts\python.exe src\etapa0_preparacao_dos_dados\preparar_conversas_de_chat.py --out data\sft ^
    --smoltalk-config smol-magpie-ultra,everyday-conversations ^
    --smoltalk-weight 8 --gsm8k-weight 1 --max-tokens 45e6 --val-tokens 1.5e6
```

| | tokens | conversas/docs | tempo |
|---|---|---|---|
| Pré-treino (FineWeb-Edu `sample-10BT`) | 1.15 B + 5 M de validação | 12 shards | 196 s |
| Mid-training (SmolTalk `all` + GSM8K) | 160 M + 2 M | 181 907 conversas | 79 s |
| SFT (SmolTalk `smol-magpie-ultra` + `everyday-conversations` + GSM8K) | 45 M + 1.5 M | 34 629 conversas | 32 s |

O FineWeb-Edu é lido em *streaming*: o parquet bruto nunca toca o disco, só os
shards tokenizados (~2 bytes/token). Os primeiros 5 M tokens viram `val.npy` e
não entram em nenhum shard de treino — é sobre eles que a perplexidade é
reportada.

**Por que esse mix.** O mid-training usa o SmolTalk `all`, a mistura ampla, para
expor o modelo à maior variedade possível de formatos conversacionais logo após
o pré-treino. O SFT usa um recorte diferente e curado: `smol-magpie-ultra` (o
subconjunto de instruções de maior qualidade) combinado com
`everyday-conversations` — este último traz turnos curtos de conversa cotidiana,
que é o regime em que um modelo de 55 M efetivamente consegue responder bem e o
que a demo de chat mais exercita. O GSM8K entra nas duas etapas, na proporção de
1 para 8, pela capacidade de raciocínio em múltiplos passos.

### Etapa 1 — pré-treino

```bash
.venv\Scripts\python.exe src\etapa1_pretreino\treinar_pretreino.py
```

Para dividir em blocos (o treino todo leva ~23 h):

```bash
.venv\Scripts\python.exe src\etapa1_pretreino\treinar_pretreino.py --max-hours 6
.venv\Scripts\python.exe src\etapa1_pretreino\treinar_pretreino.py --max-hours 6 --resume
```

O `--resume` restaura pesos, estado do AdamW, step, contagem de tokens **e a
posição exata no stream de dados**, então nenhum token é visto duas vezes.

### Etapa 2 — mid-training

```bash
.venv\Scripts\python.exe src\etapa2_midtraining\treinar_midtraining.py
```

Inicializa de `checkpoints/pretrain_final.pt` (só os pesos, sem estado do
otimizador), LR 2e-4 com warmup de 50 steps, ~150 M tokens de SmolTalk + GSM8K
já no template de chat com sequence packing. A loss cobre todos os tokens.

### Etapa 3 — SFT

```bash
.venv\Scripts\python.exe src\etapa3_sft\treinar_sft.py --epochs 2
```

Inicializa do mid-training, LR 2e-5. Aqui entra a **loss mask**: os tokens de
prompt (`User: ...`) viram `-100` e são ignorados pela cross-entropy; só os
tokens da resposta do assistente contribuem para o gradiente.

### Etapa 4 — avaliação

```bash
.venv\Scripts\python.exe src\etapa4_avaliacao\avaliar_perplexidade_e_benchmarks.py --ckpt checkpoints\pretrain_final.pt
.venv\Scripts\python.exe src\etapa4_avaliacao\avaliar_perplexidade_e_benchmarks.py --ckpt checkpoints\sft_final.pt
.venv\Scripts\python.exe src\ferramentas\plotar_curvas_de_loss.py
```

Reporta perplexidade no split de validação do FineWeb-Edu e acurácia em
HellaSwag e ARC-Easy por log-likelihood (também disponíveis: `arc_challenge`,
`piqa`, `winogrande`). Para cada alternativa somamos log P(continuação | contexto)
e escolhemos a maior — sem geração autorregressiva. Reportamos `acc` e
`acc_norm` (normalizada pelo comprimento da alternativa, que corrige o viés a
favor de respostas curtas).

### Demo em modo chat

```bash
.venv\Scripts\streamlit.exe run src\etapa5_demo\portal_de_chat.py
```

Portal de conversa para a apresentação ao vivo. Carrega o checkpoint final do
SFT e expõe temperatura, top-k, máximo de tokens e quantos turnos de histórico
entram no prompt. A resposta aparece token a token (consome
`model.generate_iter`) e cada uma vem com a contagem de tokens e a velocidade
medida — em torno de 90 tokens/s na RTX 3050.

A lista de checkpoints inclui as três etapas, o que permite mostrar a diferença
na hora: fazendo a mesma pergunta, o **pré-treino** continua o texto como se
fosse uma página da web em vez de responder, o **mid-training** já segue o
formato de pergunta e resposta, e o **SFT final** é o modelo da entrega.

O modelo foi treinado só em inglês (FineWeb-Edu + SmolTalk), então as perguntas
precisam ser em inglês. O tema da página está em `.streamlit/config.toml`.

### Slides da apresentação

```bash
.venv\Scripts\python.exe src\ferramentas\gerar_slides_da_apresentacao.py
```

Gera `Apresentacao_SLM_Projeto2.pptx` (12 slides, com notas do apresentador). A
curva de perplexidade sai de `logs/pretrain.csv` e as curvas de loss são a imagem
do `plotar_curvas_de_loss.py`, então rodar as duas etapas na ordem mantém os slides em dia
com os resultados.

O PDF entregue (`Apresentacao_SLM_Projeto2.pdf`, item 6 do enunciado) é a
exportação desse mesmo arquivo pelo PowerPoint.

## 5. Resultados

Reproduzível com `avaliar_perplexidade_e_benchmarks.py`; resultados brutos em `logs/eval_*.json`.

### Treino

| Etapa | Tempo | Tokens | Val loss | Perplexidade |
|---|---|---|---|---|
| Pré-treino | 19h50 | 1,10B | 3,549 | 34,78 (FineWeb-Edu) |
| Mid-training | 2h44 | 150M | 1,931 | 6,90 (conversacional) |
| SFT | 1h41 | 90M (2 épocas) | 1,942 | 6,97 (conversacional) |

> Os valores desta tabela são os registrados **durante o treino**, no fim de cada
> etapa (`logs/*.csv`), e é por isso que o pré-treino aparece com 34,78 aqui e com
> 36,5 nas tabelas seguintes: a avaliação durante o treino percorre só um
> subconjunto do split (`--eval-batches`, 40 no pré-treino e 30 nas etapas
> seguintes, para não roubar tempo de GPU), enquanto `avaliar_perplexidade_e_benchmarks.py` percorre o
> split inteiro. É a mesma medida em amostras de tamanhos diferentes. Daqui em
> diante, toda perplexidade de texto geral vem de `avaliar_perplexidade_e_benchmarks.py`, para que os
> números sejam comparáveis entre si.

### Benchmarks de múltipla escolha (log-verossimilhança)

Avaliação completa, sem amostragem: todos os exemplos do split indicado.

| Benchmark | n | Aleatório | Pré-treino | Mid-training | SFT |
|---|---|---|---|---|---|
| ARC-Easy (acc) | 2376 | 0,250 | **0,428** | 0,380 | 0,393 |
| PIQA (acc) | 1838 | 0,500 | **0,587** | 0,576 | 0,573 |
| WinoGrande (acc) | 1267 | 0,500 | 0,521 | **0,537** | 0,530 |
| HellaSwag (acc_norm) | 10042 | 0,250 | 0,266 | 0,269 | **0,272** |
| ARC-Challenge (acc_norm) | 1172 | 0,250 | 0,224 | 0,228 | 0,230 |
| PPL FineWeb-Edu | — | — | **36,5** | 51,3 | 54,3 |

> A coluna SFT refere-se ao checkpoint final, treinado com os dados corrigidos
> (ver "Duas falhas de preparação de dados"). A correção **não moveu os
> benchmarks**: todas as diferenças em relação ao checkpoint anterior ficaram
> abaixo de 0,006, dentro do ruído. Era o esperado — as falhas afetavam
> artefatos de geração, não o conhecimento medido por log-verossimilhança.

**Leitura honesta dos resultados.** Apenas ARC-Easy (0,428 vs 0,250) e PIQA
(0,587 vs 0,500) ficam claramente acima do acaso. HellaSwag (0,271) e WinoGrande
(0,531) estão a poucos pontos do aleatório, ou seja, praticamente sem sinal.
ARC-Challenge (0,224) fica *abaixo* do aleatório — isso não indica conhecimento
invertido, e sim ausência de conhecimento somada ao viés de comprimento das
alternativas, que empurra a escolha de forma sistemática.

Esse perfil é o esperado nesta escala: HellaSwag exige senso comum que, na
literatura, só começa a sair do chão acima de ~1B parâmetros. O resultado não
indica falha do pipeline de avaliação — se a pontuação por log-verossimilhança
estivesse incorreta, ARC-Easy e PIQA também ficariam em 0,25 e 0,50.

**Custo da especialização.** Mid-training e SFT *pioraram* ARC-Easy
(0,428 → 0,380 → 0,393) e a perplexidade em texto geral (36,5 → 51,3 → 54,3),
enquanto melhoraram marginalmente HellaSwag e WinoGrande. O modelo trocou
conhecimento factual por aderência ao formato de diálogo. Essa troca só é
visível porque avaliamos os três checkpoints; avaliando apenas o SFT, ela
passaria despercebida.

A perplexidade do pré-treino é medida no split de validação do FineWeb-Edu
(`data/pretrain/val.npy`), separado antes de gerar os shards de treino e nunca
visto durante o treinamento. As etapas seguintes reportam perplexidade nos
próprios dados conversacionais **e** nesse mesmo split do FineWeb-Edu, para
tornar visível o quanto o modelo troca capacidade de modelar texto geral por
aderência ao formato de diálogo.

### Trade-off medido no mid-training

| Split de validação | Após pré-treino | Após mid-training | Após SFT |
|---|---|---|---|
| Conversacional | — | **6,90** | **6,97** |
| FineWeb-Edu (texto geral) | **36,53** | **51,30** | **54,30** |

A linha do FineWeb-Edu vem toda de `avaliar_perplexidade_e_benchmarks.py`, para que as três colunas sejam
comparáveis entre si. A linha conversacional só existe durante o treino (é o
split de validação de cada etapa), e vem de `logs/midtrain.csv` e `logs/sft.csv`.

O mid-training derrubou a perplexidade no formato de diálogo e **piorou** a de
texto geral, de 36,5 para 51,3 (+40%). É esquecimento parcial do pré-treino,
esperado ao especializar o modelo, e só fica visível porque avaliamos os dois
splits em cada etapa. Não tratamos isso como defeito: o objetivo da etapa é o
formato conversacional. Vale registrar como o custo explícito da especialização.

O SFT move pouco as duas perplexidades (6,90 → 6,97 no conversacional e
51,3 → 54,3 em texto geral). Isso é esperado e não indica falha: com LR 2e-5,
duas épocas e loss mask, a etapa ajusta o comportamento de resposta, não a
capacidade de modelagem de linguagem. A perplexidade do SFT também não é
comparável de forma direta às anteriores, porque só os tokens do assistente
entram na loss. O efeito
do SFT aparece na qualidade das respostas (seção de demonstrações), não aqui.

### Evolução da perplexidade no pré-treino

Medições feitas durante o treino (`logs/pretrain.csv`), então na mesma escala da
tabela "Treino" acima — 34,8 no fim, e não os 36,5 do split completo.

| Step | Tokens | Val PPL |
|---|---|---|
| 200 | 52M | 306,0 |
| 400 | 105M | 134,1 |
| 600 | 157M | 91,2 |
| 2800 | 734M | 38,0 |
| 3400 | 891M | 36,0 |
| 4196 | 1,10B | **34,8** |

Nos últimos 20% do treino a perplexidade caiu apenas 3,3% (36,0 → 34,8) ao custo
de 200M tokens. O retorno decrescente nessa faixa indica que o gargalo passou a
ser a capacidade do modelo, não a quantidade de dados — o que é consistente com
1,0× Chinchilla ser o ponto de parada adequado para 55M parâmetros. Estender o
treino sem aumentar o modelo renderia pouco; o ganho viria de mais parâmetros.

Treino executado em 19h50 numa RTX (6 GB), throughput médio de 15,4K tokens/s.

Curvas de loss: `logs/curvas_loss.png` (gerado por `plotar_curvas_de_loss.py`).

### Demonstrações de geração

Amostras completas em `logs/samples_*.json`, geradas com `gerar_amostras_de_texto.py`
(T=0,8, top-k=50).

**Antes vs. depois do SFT** (mesma pergunta, `midtrain_final` vs `sft_final`):

| | Antes do SFT | Depois do SFT |
|---|---|---|
| Tom | expositivo, sem registro de resposta | registro de assistente ("Here are a few tips") |
| Estrutura | listas numeradas que se perdem | parágrafos e listas coerentes |
| Poema | prosa repetitiva sobre o tema | texto com imagética e ritmo |
| Fatos | errados | **igualmente errados** |
| Aritmética | errada | **errada** |

O SFT melhorou forma, não conteúdo. É o resultado correto a esperar: duas épocas
a LR 2e-5 ajustam comportamento de resposta, não adicionam conhecimento. Ambos
os checkpoints afirmam que a capital da França é Paris e, na frase seguinte,
"Saint-August, France".

### Duas falhas de preparação de dados encontradas e corrigidas

As primeiras amostras geradas após o SFT expuseram dois problemas na nossa
preparação dos dados conversacionais. Ambos foram corrigidos em
`prepare_chat.py` e os dados reprocessados.

**1. Anotações de calculadora do GSM8K vazavam para a saída.** Perguntado
"If I have 3 apples and buy 7 more, how many do I have?", o modelo respondia:

```
There are 7 * 7 = <<7*7=90>>90 apples on a second day.
...
A resposta e: 92
```

As sequências `<<7*7=90>>` são sintaxe de calculadora usada na construção do
GSM8K, não parte da resposta. Mantidas no corpus, o modelo aprendeu a emitir um
formato que não tem como executar — e, como se vê, o conteúdo dentro da anotação
está aritmeticamente errado.

**2. Mistura de idiomas introduzida por nós.** Substituímos o marcador `####` do
GSM8K por `"A resposta e:"`, injetando português num modelo treinado
exclusivamente em inglês. Deveria ter sido `"The answer is:"`.

**Correção.** `iter_gsm8k()` passou a aplicar `re.sub(r"<<[^>]*>>", "", ...)` e a
usar `"The answer is:"`. O filtro é aplicado **somente ao GSM8K**, de propósito:
conversas de código do SmolTalk contêm `std::cout << ... >>` legitimamente, e um
filtro global destruiria esses exemplos.

**Verificação nos dados regravados** (amostra de 8M tokens por conjunto,
contando o padrão restrito `<<[\d\s()+*/.,=-]+>>`):

| Conjunto | Anotações de calculadora | `"A resposta e:"` | `"The answer is:"` |
|---|---|---|---|
| `data/sft` | **0** | **0** | 605 |
| `data/midtrain` | 22 | **0** | 1150 |

As 22 ocorrências remanescentes no mid-training **não vêm do nosso loader**: o
config `all` do SmolTalk inclui conversas de matemática com a anotação já
embutida na fonte. A 22 ocorrências por 8M tokens (0,0003%), não compensa
filtrar de forma agressiva e arriscar corromper exemplos de código.

Nenhum dos dois problemas invalidava os resultados de perplexidade ou dos
benchmarks de múltipla escolha, que não dependem de geração autoregressiva —
afetavam a qualidade qualitativa das respostas e a demo.

**Verificação na saída do modelo** (`src/etapa4_avaliacao/verificar_vazamento_do_gsm8k.py`). Comparar os dados não
basta: o que importa é se o modelo ainda emite os artefatos. O script gera 32
respostas por checkpoint (8 prompts aritméticos × 4 amostras, porque a geração é
estocástica e uma amostra limpa não prova ausência) e conta as ocorrências:

| Checkpoint | Amostras | `<<calculadora>>` | `"A resposta e"` |
|---|---|---|---|
| `sft_final_v1.pt` (dados sujos) | 32 | **54** | **16** |
| `sft_final.pt` (dados limpos) | 32 | **0** | **0** |

A correção aplicada apenas ao SFT foi suficiente: o vazamento desapareceu por
completo, sem necessidade de repetir o mid-training (economia de ~2h44 de GPU).
A linha de base do v1 é o que torna o zero do v2 interpretável — sem ela, um
resultado nulo não distinguiria "a correção funcionou" de "os prompts não
ativaram o comportamento".

Exemplo da mesma pergunta nos dois checkpoints:

```
v1: You have 4 apples and 2 apples, so you have 5 * 2 = <<5*2=6>>6 apples.
    ...
    A resposta e: 36

v2: If I have 3 apples and buy 7 more than 3 times that, then each fruit will
    be sold to my local farmer, with a total of 7 + 1 = 10 apples.
    The answer is: 40
```

A aritmética continua errada nos dois — isso é a limitação de 55M parâmetros, não
o problema de dados. O que mudou é que o v2 não emite mais sintaxe de calculadora
nem mistura idiomas.

O checkpoint treinado antes da correção está preservado como `sft_final_v1.pt`,
com seus logs e amostras em `logs/*_v1.*`, para reproduzir a comparação.

## 6. Checkpoints

Não versionados no git — ficam no HuggingFace Hub:

**https://huggingface.co/Floriano03/Trabalho_SML_Profundo2**

| Arquivo | Etapa | Tamanho |
|---|---|---|
| `pretrain_final.pt` | pré-treino | 211 MB |
| `midtrain_final.pt` | mid-training | 211 MB |
| `sft_final.pt` | **SFT — o modelo final** | 211 MB |

Cada arquivo guarda só `model` e `config`. O estado do otimizador fica de fora —
é o que torna os `_best`/`_last` locais três vezes maiores, e ele não serve nem
para avaliar nem para a demo.

Para baixar os três para `checkpoints/`, que é onde `avaliar_perplexidade_e_benchmarks.py`, `gerar_amostras_de_texto.py`
e a demo procuram:

```bash
.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download as d; [d('Floriano03/Trabalho_SML_Profundo2', f, local_dir='checkpoints') for f in ['pretrain_final.pt','midtrain_final.pt','sft_final.pt']]"
```

## 7. Créditos

Código do pipeline de treino, avaliação e demo escrito para este projeto. Nos
inspiramos nas seguintes implementações de referência, sem copiar código:

- **nanoGPT** / **build-nanoGPT**, Andrej Karpathy (MIT) — organização dos shards
  de dados, truque de arredondar o vocab para múltiplo de 64 e init escalado nas
  projeções residuais.
- **LLaMA** (Touvron et al., 2023) — RoPE + RMSNorm + SwiGLU como bloco base.
- Pré-processamento canônico de HellaSwag/ARC seguindo o `lm-evaluation-harness`
  (EleutherAI), reimplementado aqui.

### Referências

1. Vaswani et al. (2017), *Attention Is All You Need*
2. Brown et al. (2020), *Language Models are Few-Shot Learners*
3. Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models*
4. Touvron et al. (2023), *LLaMA: Open and Efficient Foundation Language Models*
5. Su et al. (2022), *RoFormer: Enhanced Transformer with Rotary Position Embedding*
6. Shazeer (2020), *GLU Variants Improve Transformer*
7. Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models*
8. HuggingFace (2024), *SmolLM*
