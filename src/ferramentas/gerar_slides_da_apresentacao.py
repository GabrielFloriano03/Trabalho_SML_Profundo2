"""Gera a apresentacao do Projeto 2 (SLM) a partir dos resultados em logs/.

A curva de perplexidade do slide 4 e lida de `logs/pretrain.csv` e as curvas de
loss sao a imagem gerada por `plotar_curvas_de_loss.py`. Os demais numeros estao escritos
aqui, conferidos contra `logs/eval_*.json` e os CSVs — nenhum e estimado.

Depende de `python-pptx`, que nao faz parte do pipeline de treino.

Uso (a partir da raiz do projeto):
    python src/ferramentas/gerar_slides_da_apresentacao.py
    python src/ferramentas/gerar_slides_da_apresentacao.py --out outro.pptx
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_TICK_MARK
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--root", default=".", help="raiz do projeto (onde esta logs/)")
ap.add_argument("--out", default="Apresentacao_SLM_Projeto2.pptx")
args = ap.parse_args()

ROOT = Path(args.root)
OUT = Path(args.out)

# ---------------------------------------------------------------- paleta
INK = RGBColor(0x1C, 0x21, 0x27)     # carvao: fundo escuro e texto no claro
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TINT = RGBColor(0xF1, 0xF3, 0xF5)    # cartao claro
TINT_DK = RGBColor(0x2A, 0x31, 0x39)  # cartao no slide escuro
MUTED = RGBColor(0x5A, 0x66, 0x72)
MUTED_DK = RGBColor(0x9A, 0xA3, 0xAC)
AMBER = RGBColor(0xE0, 0x8A, 0x2E)   # acento unico
AMBER_LT = RGBColor(0xF2, 0xA5, 0x41)
TEAL = RGBColor(0x2F, 0x6F, 0x6E)
BRICK = RGBColor(0xB3, 0x41, 0x2C)
GRAY = RGBColor(0xB6, 0xBE, 0xC6)

HEAD = "Cambria"
BODY = "Calibri"
MONO = "Courier New"

M = 0.65                  # margem lateral
W = 13.333 - 2 * M        # largura util
CONTENT_Y = 1.62          # topo do conteudo, igual em todo slide de conteudo

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


# ------------------------------------------------------------- helpers
def slide(dark=False):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = INK if dark else WHITE
    return s


def tb(s, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT):
    """paras: lista de dicts {t, sz, b, i, c, font, space_after, space_before, line}."""
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = p.get("align", align)
        if p.get("space_after"):
            para.space_after = Pt(p["space_after"])
        if p.get("space_before"):
            para.space_before = Pt(p["space_before"])
        if p.get("line"):
            para.line_spacing = p["line"]
        # "runs" permite estilos diferentes na mesma linha, sem posicionar duas
        # caixas lado a lado (o que exigiria adivinhar a largura do texto)
        for r in p.get("runs", [p]):
            run = para.add_run()
            run.text = r["t"]
            f = run.font
            f.size = Pt(r.get("sz", p.get("sz", 14)))
            f.bold = r.get("b", False)
            f.italic = r.get("i", False)
            f.name = r.get("font", p.get("font", BODY))
            f.color.rgb = r.get("c", INK)
    return box


def title(s, text, dark=False):
    tb(s, M, 0.52, W, 0.9,
       [{"t": text, "sz": 34, "b": True, "font": HEAD, "c": WHITE if dark else INK}])


def card(s, x, y, w, h, fill=TINT, radius=0.035):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                             Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    shp.shadow.inherit = False
    shp.adjustments[0] = radius
    shp.text_frame.text = ""
    return shp


def chip(s, x, y, d, text, fill=AMBER_LT, fg=INK, sz=14):
    """Motivo visual que se repete: numero/icone dentro de um circulo ambar."""
    shp = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.size = Pt(sz)
    r.font.bold = True
    r.font.name = BODY
    r.font.color.rgb = fg
    return shp


def style_chart(chart, cat_sz=11, val_sz=11, dark=False):
    lbl = MUTED_DK if dark else MUTED
    chart.font.size = Pt(cat_sz)
    chart.font.name = BODY
    chart.font.color.rgb = lbl
    chart.has_title = False
    chart.has_legend = False
    for ax in (chart.category_axis, chart.value_axis):
        ax.has_major_gridlines = False
        ax.has_minor_gridlines = False
        ax.major_tick_mark = XL_TICK_MARK.NONE
        ax.minor_tick_mark = XL_TICK_MARK.NONE
        ax.tick_labels.font.size = Pt(val_sz)
        ax.tick_labels.font.name = BODY
        ax.tick_labels.font.color.rgb = lbl
        ax.format.line.color.rgb = GRAY if not dark else MUTED
    chart.value_axis.has_major_gridlines = True
    gl = chart.value_axis.major_gridlines.format.line
    gl.color.rgb = RGBColor(0xE4, 0xE8, 0xEB) if not dark else RGBColor(0x3A, 0x42, 0x4A)
    gl.width = Pt(0.75)


def table(s, x, y, w, rows, col_w, header_fill=INK, sz=12, head_sz=12, row_h=0.42):
    n_r, n_c = len(rows), len(rows[0])
    shp = s.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w),
                             Inches(row_h * n_r))
    t = shp.table
    t.first_row = False
    t.horz_banding = False
    total = sum(col_w)
    for j, cw in enumerate(col_w):
        t.columns[j].width = Emu(int(Inches(w) * cw / total))
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, cell in enumerate(row):
            txt, bold, color = (cell if isinstance(cell, tuple) else (cell, False, None))
            c = t.cell(i, j)
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = header_fill
            else:
                c.fill.fore_color.rgb = WHITE if i % 2 else TINT
            c.margin_left = c.margin_right = Inches(0.11)
            c.margin_top = c.margin_bottom = Inches(0.04)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = txt
            r.font.size = Pt(head_sz if i == 0 else sz)
            r.font.name = BODY
            r.font.bold = bold or i == 0
            r.font.color.rgb = (WHITE if i == 0 else (color or INK))
    return t


# ================================================== 1. capa
s = slide(dark=True)
tb(s, M, 1.45, 10.6, 2.3, [
    {"t": "Small Language Model\ntreinado do zero", "sz": 46, "b": True,
     "font": HEAD, "c": WHITE, "line": 1.05},
])
tb(s, M, 3.62, 11.0, 0.45, [
    {"t": "Pré-treino  →  Mid-training  →  SFT  →  Avaliação & demo",
     "sz": 19, "b": True, "c": AMBER_LT},
])
tb(s, M, 4.18, 10.6, 0.4, [
    {"t": "Decoder-only estilo LLaMA, treinado inteiramente em uma GPU de notebook",
     "sz": 15, "c": MUTED_DK},
])

stats = [("55,26 M", "parâmetros treináveis"), ("1,10 B", "tokens de pré-treino"),
         ("24 h", "de GPU nas três etapas"), ("6 GB", "de VRAM disponíveis")]
cw = (W - 3 * 0.3) / 4
for i, (big, lab) in enumerate(stats):
    x = M + i * (cw + 0.3)
    card(s, x, 5.1, cw, 1.1, fill=TINT_DK)
    tb(s, x + 0.25, 5.26, cw - 0.5, 0.5,
       [{"t": big, "sz": 26, "b": True, "font": HEAD, "c": AMBER_LT}])
    tb(s, x + 0.25, 5.78, cw - 0.5, 0.35, [{"t": lab, "sz": 11, "c": MUTED_DK}])

tb(s, M, 6.68, W, 0.35, [
    {"t": "Aprendizado Profundo 2  ·  Projeto 2  ·  Gabriel Floriano  ·  PUCRS  ·  novembro de 2026",
     "sz": 11, "c": MUTED_DK}])
s.notes_slide.notes_text_frame.text = (
    "O objetivo nao foi usar um modelo pronto: foi executar o pipeline completo de um LLM "
    "em escala reduzida, com as quatro etapas em sequencia, dentro de 6 GB de VRAM. "
    "Tudo que vem a seguir foi medido nesta maquina."
)

# ================================================== 2. pipeline
s = slide()
title(s, "Quatro etapas, uma GPU de notebook")
tb(s, M, 1.32, W, 0.35, [
    {"t": "Cada etapa inicializa dos pesos da anterior — sem estado do otimizador, como pede o enunciado",
     "sz": 13, "c": MUTED}])

etapas = [
    ("01", "Pré-treino", "FineWeb-Edu (sample-10BT)",
     [("Tokens", "1,10 B  ·  1,0× Chinchilla"), ("Tempo", "19 h 50"),
      ("LR", "6e-4, cosseno"), ("Val loss", "3,549  →  PPL 34,8")]),
    ("02", "Mid-training", "SmolTalk all + GSM8K",
     [("Tokens", "150 M"), ("Tempo", "2 h 44"),
      ("LR", "2e-4, warmup 50"), ("Val loss", "1,931  →  PPL 6,90")]),
    ("03", "SFT", "SmolTalk curado + GSM8K",
     [("Tokens", "90 M  ·  2 épocas"), ("Tempo", "1 h 41"),
      ("LR", "2e-5 + loss mask"), ("Val loss", "1,942  →  PPL 6,97")]),
    ("04", "Avaliação & demo", "log-verossimilhança + Streamlit",
     [("Benchmarks", "5, sem amostragem"), ("Checkpoints", "os 3 avaliados"),
      ("Perplexidade", "split separado"), ("Demo", "chat com sft_final")]),
]
cw = (W - 3 * 0.32) / 4
for i, (num, nome, fonte, linhas) in enumerate(etapas):
    x = M + i * (cw + 0.32)
    card(s, x, 1.95, cw, 3.95)
    chip(s, x + 0.28, 2.2, 0.46, num)
    tb(s, x + 0.28, 2.82, cw - 0.56, 0.42,
       [{"t": nome, "sz": 17, "b": True, "font": HEAD}])
    tb(s, x + 0.28, 3.26, cw - 0.56, 0.6,
       [{"t": fonte, "sz": 11, "c": MUTED, "line": 1.1}])
    paras = []
    for k, v in linhas:
        paras.append({"t": k, "sz": 10, "c": MUTED, "space_before": 7})
        paras.append({"t": v, "sz": 12.5, "b": True})
    tb(s, x + 0.28, 3.95, cw - 0.56, 1.8, paras)

tb(s, M, 6.1, W, 0.8, [
    {"t": "Por que mid-training entre o pré-treino e o SFT", "sz": 14, "b": True,
     "space_after": 4},
    {"t": "O pré-treino só viu texto corrido. O mid-training apresenta o template de "
          "conversa em volume (150 M tokens, mistura ampla) antes de o SFT refinar o "
          "comportamento de resposta com loss mask e LR dez vezes menor em dados curados.",
     "sz": 13, "c": MUTED, "line": 1.25},
])
s.notes_slide.notes_text_frame.text = (
    "As quatro etapas sao sequenciais e cada uma parte dos pesos da anterior. "
    "Repare na queda da perplexidade: 34,8 no pre-treino para 6,90 no mid-training. "
    "Sao numeros em splits diferentes, nao comparaveis entre si - volto nisso no slide do trade-off."
)

# ================================================== 3. arquitetura
s = slide()
title(s, "Arquitetura: decoder-only estilo LLaMA")

card(s, M, CONTENT_Y, 4.25, 4.9)
tb(s, M + 0.35, CONTENT_Y + 0.35, 3.55, 1.5, [
    {"t": "55,26 M", "sz": 40, "b": True, "font": HEAD, "c": AMBER},
    {"t": "parâmetros treináveis", "sz": 13, "b": True, "space_before": 2},
    {"t": "29,50 M sem contar embeddings", "sz": 11, "c": MUTED, "space_before": 1},
])
tb(s, M + 0.35, CONTENT_Y + 1.95, 3.55, 2.5, [
    {"t": "10 camadas", "sz": 12.5, "font": MONO, "space_after": 5},
    {"t": "d_model      512", "sz": 12.5, "font": MONO, "space_after": 5},
    {"t": "heads      8 Q / 4 KV", "sz": 12.5, "font": MONO, "space_after": 5},
    {"t": "ffn_hidden  1408", "sz": 12.5, "font": MONO, "space_after": 5},
    {"t": "contexto    1024", "sz": 12.5, "font": MONO, "space_after": 5},
    {"t": "vocab      50304 (BPE GPT-2)", "sz": 12.5, "font": MONO, "space_after": 5},
    {"t": "precisão    bf16 autocast", "sz": 12.5, "font": MONO},
])

escolhas = [
    ("RoPE", "θ = 10000",
     "Codifica posição relativa dentro da própria atenção: não gasta parâmetros com "
     "uma tabela de embeddings e extrapola melhor que o posicional aprendido do GPT-2."),
    ("GQA", "8 query / 4 KV heads",
     "Corta metade dos parâmetros de K e V e reduz o KV-cache pela metade na geração — "
     "é o que viabiliza a demo com contexto cheio em 6 GB."),
    ("SwiGLU", "hidden 1408 ≈ 8/3 · d_model",
     "Melhor qualidade por FLOP que GELU; o fator 8/3 compensa a terceira matriz do "
     "gate e mantém a contagem equivalente a um MLP 4×."),
    ("RMSNorm", "pre-norm",
     "Cerca de 15% mais barata que LayerNorm por não centrar, e o pre-norm mantém o "
     "caminho residual limpo — treino estável sem warmup longo."),
    ("Weight tying", "entrada = saída",
     "Economiza 25,7 M parâmetros, que foram realocados para as camadas do transformer."),
]
x2 = M + 4.25 + 0.55
w2 = W - 4.25 - 0.55
for i, (nome, det, desc) in enumerate(escolhas):
    y = CONTENT_Y + i * 0.99
    chip(s, x2, y + 0.05, 0.34, "·", sz=16)
    tb(s, x2 + 0.5, y, w2 - 0.5, 0.3, [
        {"runs": [{"t": nome, "sz": 15, "b": True, "font": HEAD},
                  {"t": "   " + det, "sz": 11, "b": True, "c": AMBER}]},
    ])
    tb(s, x2 + 0.5, y + 0.3, w2 - 0.5, 0.62, [
        {"t": desc, "sz": 12, "c": MUTED, "line": 1.18},
    ])

tb(s, M, 6.68, W, 0.35, [
    {"t": "Cada componente é uma variante moderna do transformer de 2017, escolhida por "
          "um motivo que cabe nas restrições deste projeto — não por ser recente.",
     "sz": 12, "i": True, "c": MUTED}])
s.notes_slide.notes_text_frame.text = (
    "Nenhuma dessas escolhas e gratuita. A que mais importou foi GQA: sem reduzir o "
    "KV-cache, a demo de chat com contexto de 1024 tokens nao caberia nos 6 GB. "
    "Weight tying devolveu 25,7 M parametros para as camadas."
)

# ================================================== 4. scaling laws
s = slide()
title(s, "Por que exatamente 1,1 bilhão de tokens")

card(s, M, CONTENT_Y, 4.25, 2.35)
tb(s, M + 0.35, CONTENT_Y + 0.28, 3.55, 1.8, [
    {"t": "Chinchilla (Hoffmann et al., 2022)", "sz": 11, "c": MUTED, "space_after": 7},
    {"t": "20 tokens por parâmetro", "sz": 15, "b": True, "space_after": 9},
    {"t": "55,26 M × 20 = 1,105 B", "sz": 15, "b": True, "font": MONO, "c": AMBER,
     "space_after": 7},
    {"t": "Arredondado para 1,1 B: o ponto compute-ótimo, não um compromisso de prazo.",
     "sz": 12, "c": MUTED, "line": 1.2},
])

card(s, M, CONTENT_Y + 2.55, 4.25, 2.35)
tb(s, M + 0.35, CONTENT_Y + 2.83, 3.55, 1.8, [
    {"t": "O custo foi medido antes de começar", "sz": 11, "c": MUTED, "space_after": 7},
    {"t": "13 564 tokens/s = 48,8 M/h", "sz": 13, "b": True, "font": MONO,
     "space_after": 7},
    {"t": "1,1 B ÷ 48,8 M ≈ 22,6 h", "sz": 13, "b": True, "font": MONO,
     "c": AMBER, "space_after": 7},
    {"t": "Previstas antes de começar. Executado em 19 h 50, a 15,4 K tokens/s de média.",
     "sz": 12, "c": MUTED, "line": 1.2},
])

# curva de perplexidade de validacao, direto do CSV do pre-treino
pts = []
with open(ROOT / "logs" / "pretrain.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("val_ppl"):
            pts.append((int(row["tokens"]) / 1e9, float(row["val_ppl"])))

cd = XyChartData()
ser = cd.add_series("PPL de validação")
for xv, yv in pts:
    ser.add_data_point(xv, yv)
cx, cy, cwid, chei = M + 4.8, CONTENT_Y - 0.08, W - 4.8, 3.55
gf = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS,
                        Inches(cx), Inches(cy), Inches(cwid), Inches(chei), cd)
ch = gf.chart
style_chart(ch)
ch.value_axis.maximum_scale = 320
ch.value_axis.minimum_scale = 0
ch.category_axis.maximum_scale = 1.1
ch.category_axis.minimum_scale = 0
sr = ch.plots[0].series[0]
sr.format.line.color.rgb = AMBER
sr.format.line.width = Pt(2.5)
tb(s, cx + 0.1, cy + chei - 0.02, cwid - 0.2, 0.3,
   [{"t": "perplexidade no split de validação  ×  tokens vistos (bilhões)",
     "sz": 10.5, "c": MUTED, "align": PP_ALIGN.CENTER}])

tb(s, cx, 5.42, cwid, 1.5, [
    {"t": "O retorno decrescente confirma o ponto de parada", "sz": 14, "b": True,
     "space_after": 5},
    {"t": "Nos últimos 200 M tokens a perplexidade caiu de 36,0 para 34,8 — apenas 3,3% "
          "ao custo de 18% do orçamento. O gargalo passou a ser a capacidade do modelo, "
          "não a quantidade de dados, o que é consistente com 1,0× Chinchilla ser o ponto "
          "adequado para 55 M parâmetros. Estender o treino renderia pouco: o ganho viria "
          "de mais parâmetros.",
     "sz": 12.5, "c": MUTED, "line": 1.22},
])
s.notes_slide.notes_text_frame.text = (
    "O orcamento de tokens nao foi escolhido por conveniencia. Chinchilla da 20 tokens "
    "por parametro, o que para 55,26 M da 1,105 B - arredondamos para 1,1 B. "
    "A curva a direita e a evidencia de que esse era o ponto certo de parar: o plato final "
    "mostra que faltava capacidade, nao dados."
)

# ================================================== 5. micro-batch medido
s = slide()
title(s, "6 GB de VRAM: o micro-batch foi medido, não estimado")

mb = [("2", 12223, 2.65), ("3", 13226, 3.58), ("4", 13564, 4.52),
      ("5", 13932, 5.40), ("6", 9113, 6.32), ("8", 3808, 8.14)]
cd = CategoryChartData()
cd.categories = [m[0] for m in mb]
cd.add_series("tokens/s", [m[1] for m in mb])
cx, cy, cwid, chei = M + 4.55, CONTENT_Y - 0.05, W - 4.55, 4.05
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,
                        Inches(cx), Inches(cy), Inches(cwid), Inches(chei), cd)
ch = gf.chart
style_chart(ch)
plot = ch.plots[0]
plot.gap_width = 55
plot.has_data_labels = True
dl = plot.data_labels
dl.font.size = Pt(10.5)
dl.font.name = BODY
dl.font.color.rgb = MUTED
dl.number_format = "#,##0"
dl.number_format_is_linked = False
cores = [GRAY, GRAY, AMBER, TEAL, BRICK, BRICK]
for i, cor in enumerate(cores):
    pt = plot.series[0].points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = cor
tb(s, cx + 0.1, cy + chei - 0.05, cwid - 0.2, 0.3,
   [{"t": "throughput medido  ×  micro-batch   (medir_throughput_da_gpu.py)",
     "sz": 10.5, "c": MUTED, "align": PP_ALIGN.CENTER}])

card(s, M, CONTENT_Y, 4.25, 2.0)
tb(s, M + 0.35, CONTENT_Y + 0.3, 3.55, 1.5, [
    {"t": "micro-batch 4", "sz": 26, "b": True, "font": HEAD, "c": AMBER,
     "space_after": 5},
    {"t": "+ gradient accumulation 64", "sz": 13, "b": True, "space_after": 4},
    {"t": "= batch global de 2¹⁸ = 262 144 tokens por passo de otimização",
     "sz": 12, "c": MUTED, "line": 1.2},
])

notas = [
    ("Acima de 5 o driver pagina",
     "Em micro-batch 6 o pico ultrapassa os 6 GB físicos e o WDDM passa a paginar para "
     "a RAM do sistema: o throughput cai 3–4×, de 13,6 K para 9,1 K tokens/s."),
    ("Por que 4 e não 5",
     "O 5 é 2,6% mais rápido, mas deixa só 0,5 GB de folga para o desktop. Um pico "
     "qualquer de VRAM derrubaria 20 h de treino para ganhar meia hora."),
]
for i, (h, d) in enumerate(notas):
    y = CONTENT_Y + 2.25 + i * 1.3
    chip(s, M, y + 0.02, 0.32, "!", sz=14)
    tb(s, M + 0.47, y, 4.25 - 0.47, 0.3, [{"t": h, "sz": 13.5, "b": True, "font": HEAD}])
    tb(s, M + 0.47, y + 0.32, 4.25 - 0.47, 0.9, [{"t": d, "sz": 12, "c": MUTED,
                                                  "line": 1.2}])
s.notes_slide.notes_text_frame.text = (
    "Esse grafico e o motivo de o projeto ter cabido em 6 GB. A intuicao diz que batch "
    "maior e sempre melhor; a medicao mostra um penhasco entre 5 e 6, quando o driver "
    "comeca a paginar para a RAM. Rodar o bench antes custou minutos e evitou descobrir "
    "isso depois de horas de treino."
)

# ================================================== 6. dados
s = slide()
title(s, "Dados: três corpora, um tokenizer")

corpora = [
    ("Pré-treino", "FineWeb-Edu  ·  sample-10BT",
     [("Tokens", "1,15 B + 5 M de validação"), ("Shards", "12 arquivos uint16")],
     "Preparado em 196 s, em streaming: o parquet bruto nunca toca o disco, só os shards "
     "tokenizados a 2 bytes por token."),
    ("Mid-training", "SmolTalk all  +  GSM8K",
     [("Tokens", "160 M + 2 M"), ("Conversas", "181 907")],
     "Preparado em 79 s. A mistura ampla de propósito: expõe o modelo à maior variedade "
     "de formatos conversacionais possível."),
    ("SFT", "smol-magpie-ultra + everyday-conversations + GSM8K",
     [("Tokens", "45 M + 1,5 M"), ("Conversas", "34 629  ·  8 : 1 com GSM8K")],
     "Recorte curado, com turnos curtos de conversa — o regime em que um modelo de 55 M "
     "efetivamente responde bem."),
]
cwid = (W - 2 * 0.4) / 3
for i, (nome, fonte, linhas, nota) in enumerate(corpora):
    x = M + i * (cwid + 0.4)
    card(s, x, CONTENT_Y, cwid, 3.85)
    chip(s, x + 0.3, CONTENT_Y + 0.28, 0.42, str(i + 1))
    tb(s, x + 0.3, CONTENT_Y + 0.84, cwid - 0.6, 0.35,
       [{"t": nome, "sz": 17, "b": True, "font": HEAD}])
    tb(s, x + 0.3, CONTENT_Y + 1.22, cwid - 0.6, 0.52,
       [{"t": fonte, "sz": 11, "c": AMBER, "b": True, "line": 1.12}])
    paras = []
    for k, v in linhas:
        paras.append({"t": k, "sz": 10, "c": MUTED, "space_before": 6})
        paras.append({"t": v, "sz": 12.5, "b": True})
    tb(s, x + 0.3, CONTENT_Y + 1.80, cwid - 0.6, 0.95, paras)
    tb(s, x + 0.3, CONTENT_Y + 2.90, cwid - 0.6, 0.8,
       [{"t": nota, "sz": 11, "c": MUTED, "i": True, "line": 1.15}])

dest = [
    ("Split de validação separado antes de tudo",
     "Os primeiros 5 M tokens do FineWeb-Edu viram val.npy e não entram em nenhum shard "
     "de treino. É sobre eles que toda perplexidade de texto geral é reportada."),
    ("Loss mask no SFT",
     "Os tokens de prompt viram −100 e são ignorados pela cross-entropy: só os tokens da "
     "resposta do assistente contribuem para o gradiente."),
]
wd = (W - 0.4) / 2
for i, (h, d) in enumerate(dest):
    x = M + i * (wd + 0.4)
    tb(s, x, 5.78, wd, 0.3, [{"t": h, "sz": 13.5, "b": True, "font": HEAD}])
    tb(s, x, 6.10, wd, 0.85, [{"t": d, "sz": 12, "c": MUTED, "line": 1.22}])
s.notes_slide.notes_text_frame.text = (
    "Dois detalhes que valem a pena. Primeiro: o split de validacao foi separado antes de "
    "gerar qualquer shard de treino, entao a perplexidade e honesta. Segundo: o SFT usa "
    "loss mask - o modelo nao e premiado por reproduzir a pergunta do usuario."
)

# ================================================== 7. curvas
s = slide()
title(s, "Curvas de loss das três etapas")
img = ROOT / "logs" / "curvas_loss.png"
iw, ih = Image.open(img).size
max_w, max_h = W, 4.7
scale = min(max_w / iw, max_h / ih)
dw, dh = iw * scale, ih * scale
s.shapes.add_picture(str(img), Inches(M + (W - dw) / 2), Inches(CONTENT_Y - 0.05),
                     Inches(dw), Inches(dh))
tb(s, M, 6.45, W, 0.85, [
    {"t": "Gerado por plotar_curvas_de_loss.py a partir dos CSVs de cada etapa. A loss de "
          "treino do pré-treino cai de 10,91 — exatamente ln(50304) = 10,83, o que valida "
          "a inicialização — até 3,55. Mid-training e SFT partem de valores muito mais "
          "baixos porque herdam os pesos da etapa anterior.",
     "sz": 12.5, "c": MUTED, "line": 1.22}])
s.notes_slide.notes_text_frame.text = (
    "A loss inicial do pre-treino batendo com ln(vocab) e o primeiro sinal de que a "
    "inicializacao esta certa: o modelo comeca exatamente tao ignorante quanto deveria, "
    "distribuindo probabilidade uniforme sobre o vocabulario."
)

# ================================================== 8. benchmarks
s = slide()
title(s, "Benchmarks por log-verossimilhança")
tb(s, M, 1.32, W, 0.35, [
    {"t": "Todos os exemplos de cada split, sem amostragem. Para cada alternativa somamos "
          "log P(continuação | contexto) e escolhemos a maior — sem geração autorregressiva.",
     "sz": 12.5, "c": MUTED}])

rows = [
    ["Benchmark", "n", "Aleatório", "Pré-treino", "Mid-training", "SFT"],
    ["ARC-Easy  (acc)", "2 376", "0,250", ("0,428", True, AMBER), "0,380", "0,393"],
    ["PIQA  (acc)", "1 838", "0,500", ("0,587", True, AMBER), "0,576", "0,573"],
    ["WinoGrande  (acc)", "1 267", "0,500", "0,521", ("0,537", True, AMBER), "0,530"],
    ["HellaSwag  (acc_norm)", "10 042", "0,250", "0,266", "0,269", ("0,272", True, AMBER)],
    ["ARC-Challenge  (acc_norm)", "1 172", "0,250", ("0,224", False, BRICK), "0,228", "0,230"],
    ["Perplexidade FineWeb-Edu", "—", "—", ("36,5", True, AMBER),
     ("51,3", False, BRICK), ("54,3", False, BRICK)],
]
table(s, M, 1.95, W, rows, [3.3, 1.0, 1.2, 1.3, 1.4, 1.1], row_h=0.44)

leitura = [
    ("Acima do acaso", TEAL,
     "Só ARC-Easy (0,428 vs 0,250) e PIQA (0,587 vs 0,500) ficam claramente acima. "
     "HellaSwag e WinoGrande estão a poucos pontos do aleatório: praticamente sem sinal."),
    ("Abaixo do acaso não é erro", BRICK,
     "ARC-Challenge em 0,224 não indica conhecimento invertido, e sim ausência de "
     "conhecimento somada ao viés de comprimento das alternativas."),
    ("A avaliação está correta", AMBER,
     "Se a pontuação por log-verossimilhança estivesse errada, ARC-Easy e PIQA também "
     "ficariam em 0,25 e 0,50. Senso comum, na literatura, só sai do chão acima de ~1 B."),
]
wd = (W - 2 * 0.4) / 3
for i, (h, cor, d) in enumerate(leitura):
    x = M + i * (wd + 0.4)
    chip(s, x, 5.22, 0.3, "·", fill=cor, fg=WHITE, sz=15)
    tb(s, x + 0.44, 5.2, wd - 0.44, 0.3, [{"t": h, "sz": 13.5, "b": True, "font": HEAD}])
    tb(s, x, 5.62, wd, 1.1, [{"t": d, "sz": 12, "c": MUTED, "line": 1.22}])
s.notes_slide.notes_text_frame.text = (
    "Esse slide e deliberadamente honesto. Dois dos cinco benchmarks mostram sinal real; "
    "tres ficam no ruido. Isso e o esperado em 55 M parametros. O que valida o pipeline de "
    "avaliacao e justamente ARC-Easy e PIQA estarem acima do acaso - um bug na pontuacao "
    "deixaria todos colados na linha aleatoria."
)

# ================================================== 9. custo da especializacao
s = slide()
title(s, "O custo medido da especialização")

cd = CategoryChartData()
cd.categories = ["Após pré-treino", "Após mid-training", "Após SFT"]
# os tres valores vem da mesma execucao do script de avaliacao, comparaveis entre si
cd.add_series("PPL FineWeb-Edu", (36.53, 51.30, 54.30))
cx, cy, cwid, chei = M, CONTENT_Y, 6.0, 3.9
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,
                        Inches(cx), Inches(cy), Inches(cwid), Inches(chei), cd)
ch = gf.chart
style_chart(ch)
plot = ch.plots[0]
plot.gap_width = 90
plot.has_data_labels = True
dl = plot.data_labels
dl.font.size = Pt(13)
dl.font.bold = True
dl.font.name = BODY
dl.font.color.rgb = INK
dl.number_format = "0.0"
dl.number_format_is_linked = False
for i, cor in enumerate([TEAL, RGBColor(0xC8, 0x6A, 0x3C), BRICK]):
    pt = plot.series[0].points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = cor
tb(s, cx + 0.1, cy + chei - 0.05, cwid - 0.2, 0.3,
   [{"t": "perplexidade em texto geral (FineWeb-Edu), pelos três checkpoints  ·  "
          "avaliar_perplexidade_e_benchmarks.py",
     "sz": 10.5, "c": MUTED, "align": PP_ALIGN.CENTER}])

x2 = M + 6.0 + 0.55
w2 = W - 6.0 - 0.55
card(s, x2, CONTENT_Y, w2, 1.75, fill=TINT)
tb(s, x2 + 0.3, CONTENT_Y + 0.26, w2 - 0.6, 1.3, [
    {"t": "Ganhou formato de diálogo", "sz": 15, "b": True, "font": HEAD, "c": TEAL,
     "space_after": 5},
    {"t": "Perplexidade no split conversacional: 6,90 após o mid-training e 6,97 após "
          "o SFT. WinoGrande e HellaSwag melhoram marginalmente.",
     "sz": 12.5, "c": MUTED, "line": 1.22},
])
card(s, x2, CONTENT_Y + 1.95, w2, 1.95, fill=TINT)
tb(s, x2 + 0.3, CONTENT_Y + 2.21, w2 - 0.6, 1.5, [
    {"t": "Perdeu conhecimento factual", "sz": 15, "b": True, "font": HEAD, "c": BRICK,
     "space_after": 5},
    {"t": "Perplexidade em texto geral sobe 40% já no mid-training (36,5 → 51,3) e 49% "
          "no total. ARC-Easy cai de 0,428 para 0,380 e só volta a 0,393. É esquecimento "
          "parcial do pré-treino, esperado ao especializar.",
     "sz": 12.5, "c": MUTED, "line": 1.22},
])

tb(s, M, 5.95, W, 1.0, [
    {"t": "Essa troca só é visível porque avaliamos os três checkpoints", "sz": 15,
     "b": True, "font": HEAD, "space_after": 5},
    {"t": "Avaliando apenas o SFT — o checkpoint que a demo carrega — a degradação passaria "
          "despercebida: os números do SFT isolados não são ruins, só são piores. "
          "Não tratamos isso como defeito, e sim como o custo explícito de trocar "
          "capacidade de modelar texto geral por aderência ao formato de diálogo.",
     "sz": 12.5, "c": MUTED, "line": 1.22},
])
s.notes_slide.notes_text_frame.text = (
    "Esse e o resultado que eu mais gostei de encontrar. O mid-training e o SFT melhoram o "
    "que a gente queria - formato de dialogo - e pioram o que a gente nao estava olhando. "
    "Se tivessemos avaliado so o checkpoint final, nunca veriamos."
)

# ================================================== 10. falhas de dados
s = slide()
title(s, "Duas falhas de dados que as amostras revelaram")
tb(s, M, 1.32, W, 0.35, [
    {"t": "Nenhum teste automatizado apontou isso. Apareceu ao ler as primeiras respostas "
          "geradas depois do SFT.", "sz": 12.5, "c": MUTED}])

falhas = [
    ("1", "Anotações de calculadora do GSM8K vazavam",
     'There are 7 * 7 = <<7*7=90>>90 apples…',
     "As sequências << >> são sintaxe usada na construção do dataset, não parte da "
     "resposta. Mantidas no corpus, o modelo aprendeu a emitir um formato que não tem "
     "como executar — e o conteúdo dentro delas já vem errado."),
    ("2", "Mistura de idiomas introduzida por nós",
     'A resposta e: 92',
     'Substituímos o marcador #### do GSM8K por "A resposta e:", injetando português num '
     'modelo treinado exclusivamente em inglês. Deveria ter sido "The answer is:".'),
]
wd = (W - 0.45) / 2
for i, (num, h, code, d) in enumerate(falhas):
    x = M + i * (wd + 0.45)
    card(s, x, 1.95, wd, 2.5)
    chip(s, x + 0.3, 2.2, 0.4, num)
    tb(s, x + 0.85, 2.27, wd - 1.15, 0.3, [{"t": h, "sz": 14.5, "b": True, "font": HEAD}])
    cb = card(s, x + 0.3, 2.78, wd - 0.6, 0.5, fill=RGBColor(0xE4, 0xE8, 0xEB))
    tb(s, x + 0.45, 2.9, wd - 0.9, 0.3,
       [{"t": code, "sz": 11.5, "font": MONO, "c": BRICK}])
    tb(s, x + 0.3, 3.42, wd - 0.6, 0.95, [{"t": d, "sz": 12, "c": MUTED, "line": 1.22}])

tb(s, M, 4.68, W, 0.3, [
    {"t": "Verificação na saída do modelo  ·  verificar_vazamento_do_gsm8k.py", "sz": 14, "b": True,
     "font": HEAD}])
tb(s, M, 5.0, W, 0.3, [
    {"t": "Comparar os dados não basta: o que importa é se o modelo ainda emite os "
          "artefatos. 8 prompts aritméticos × 4 amostras, porque a geração é estocástica.",
     "sz": 12, "c": MUTED}])
rows = [
    ["Checkpoint", "Amostras", "<<calculadora>>", '"A resposta e"'],
    ["sft_final_v1.pt   (dados sujos)", "32", ("54", True, BRICK), ("16", True, BRICK)],
    ["sft_final.pt   (dados limpos)", "32", ("0", True, TEAL), ("0", True, TEAL)],
]
table(s, M, 5.42, 7.3, rows, [3.2, 1.1, 1.5, 1.4], row_h=0.4)

tb(s, M + 7.3 + 0.5, 5.42, W - 7.3 - 0.5, 1.4, [
    {"t": "O filtro é aplicado só ao GSM8K", "sz": 13.5, "b": True, "font": HEAD,
     "space_after": 4},
    {"t": "Conversas de código do SmolTalk contêm std::cout << … >> legitimamente; "
          "um filtro global destruiria esses exemplos. Corrigir apenas o SFT bastou — o "
          "vazamento desapareceu sem repetir o mid-training, economizando 2 h 44 de GPU.",
     "sz": 12, "c": MUTED, "line": 1.2},
])
s.notes_slide.notes_text_frame.text = (
    "Deixei esse slide no deck de proposito, embora seja um erro nosso. Duas licoes: "
    "ler as amostras geradas encontra coisas que nenhuma metrica agregada mostra, e a "
    "linha de base do v1 e o que torna o zero do v2 interpretavel - sem ela, um resultado "
    "nulo nao distingue 'a correcao funcionou' de 'os prompts nao ativaram o comportamento'."
)

# ================================================== 11. demo e limites
s = slide()
title(s, "A demo e os limites reais do modelo")

card(s, M, CONTENT_Y, 4.6, 4.5)
chip(s, M + 0.35, CONTENT_Y + 0.3, 0.42, "▶")
tb(s, M + 0.35, CONTENT_Y + 0.9, 4.6 - 0.7, 0.4,
   [{"t": "Demo em modo chat", "sz": 18, "b": True, "font": HEAD}])
tb(s, M + 0.35, CONTENT_Y + 1.34, 4.6 - 0.7, 0.35,
   [{"t": "streamlit run src/etapa5_demo/portal_de_chat.py", "sz": 11.5, "font": MONO, "c": AMBER}])
tb(s, M + 0.35, CONTENT_Y + 1.8, 4.6 - 0.7, 2.4, [
    {"t": "Carrega o checkpoint final do SFT, como pede o enunciado, e expõe:",
     "sz": 12.5, "c": MUTED, "line": 1.2, "space_after": 9},
    {"t": "temperatura", "sz": 13, "b": True, "space_after": 6},
    {"t": "top-k", "sz": 13, "b": True, "space_after": 6},
    {"t": "máximo de tokens gerados", "sz": 13, "b": True, "space_after": 6},
    {"t": "turnos de histórico no prompt", "sz": 13, "b": True, "space_after": 9},
    {"t": "Os outros checkpoints ficam na lista, para comparar as etapas ao vivo.",
     "sz": 12, "c": MUTED, "i": True, "line": 1.2},
])

x2 = M + 4.6 + 0.55
w2 = W - 4.6 - 0.55
listas = [
    ("O que o modelo faz", TEAL, "✓", [
        "Mantém o formato de diálogo e o registro de assistente",
        "Produz inglês fluente e gramaticalmente coerente",
        "Estrutura respostas em parágrafos e listas",
        "Segue o raciocínio passo a passo do GSM8K — a forma dele",
    ]),
    ("O que o modelo não faz", BRICK, "✕", [
        "Aritmética: erra contas de uma única operação",
        "Fatos: diz que a capital da França é Paris e, na frase seguinte, “Saint-August, France”",
        "Senso comum: HellaSwag e WinoGrande ficam no ruído",
        "Conhecimento novo — o SFT ajustou forma, não conteúdo",
    ]),
]
for i, (h, cor, mark, itens) in enumerate(listas):
    y = CONTENT_Y + i * 2.35
    chip(s, x2, y + 0.02, 0.32, mark, fill=cor, fg=WHITE, sz=13)
    tb(s, x2 + 0.46, y, w2 - 0.46, 0.35, [{"t": h, "sz": 16, "b": True, "font": HEAD}])
    paras = []
    for j, it in enumerate(itens):
        paras.append({"t": it, "sz": 12.5, "c": MUTED, "line": 1.18,
                      "space_after": 7 if j < len(itens) - 1 else 0})
    tb(s, x2 + 0.46, y + 0.42, w2 - 0.46, 1.8, paras)

tb(s, M, 6.42, W, 0.6, [
    {"t": "A aritmética errada não é um problema de dados — é o teto de 55 M parâmetros. "
          "Depois da correção o modelo parou de emitir sintaxe de calculadora, mas "
          "continua errando a conta.",
     "sz": 12.5, "i": True, "c": MUTED, "line": 1.2}])
s.notes_slide.notes_text_frame.text = (
    "Aqui eu abro a demo e mostro ao vivo. Vale pedir uma conta simples e um pedido de "
    "texto: a diferenca entre o que ele faz bem - forma - e o que ele nao faz - conteudo - "
    "fica obvia em duas perguntas."
)

# ================================================== 12. conclusao
s = slide(dark=True)
title(s, "O que o projeto mostra", dark=True)

pontos = [
    ("01", "O pipeline completo cabe em 6 GB",
     "Quatro etapas sequenciais, 1,34 B tokens processados e 24 h de GPU numa RTX 3050 "
     "de notebook. Nenhuma etapa foi pulada ou simulada."),
    ("02", "As decisões vieram de medição, não de intuição",
     "O orçamento de tokens saiu de Chinchilla, o micro-batch saiu do bench, e o ponto de "
     "parada saiu da curva de perplexidade — que mostrou retorno decrescente de 3,3%."),
    ("03", "Avaliar os três checkpoints mudou a conclusão",
     "Foi o que revelou o custo da especialização: ganho de formato conversacional, perda "
     "de 40% em perplexidade de texto geral. Só o SFT não contaria essa história."),
]
cwid = (W - 2 * 0.4) / 3
for i, (num, h, d) in enumerate(pontos):
    x = M + i * (cwid + 0.4)
    card(s, x, CONTENT_Y, cwid, 2.75, fill=TINT_DK)
    chip(s, x + 0.3, CONTENT_Y + 0.3, 0.44, num)
    tb(s, x + 0.3, CONTENT_Y + 0.92, cwid - 0.6, 0.7,
       [{"t": h, "sz": 15.5, "b": True, "font": HEAD, "c": WHITE, "line": 1.1}])
    tb(s, x + 0.3, CONTENT_Y + 1.68, cwid - 0.6, 0.95,
       [{"t": d, "sz": 12, "c": MUTED_DK, "line": 1.22}])

card(s, M, 4.82, W, 1.3, fill=TINT_DK)
tb(s, M + 0.4, 5.06, W - 0.8, 1.1, [
    {"t": "A limitação honesta", "sz": 15, "b": True, "font": HEAD, "c": AMBER_LT,
     "space_after": 5},
    {"t": "55 M parâmetros é o teto deste resultado. O modelo aprendeu a forma do "
          "raciocínio sem a capacidade de executá-lo, e três dos cinco benchmarks ficam "
          "no ruído. O caminho não é mais tokens — a curva de perplexidade mostrou que "
          "esse retorno já havia se esgotado — e sim mais parâmetros.",
     "sz": 13, "c": WHITE, "line": 1.25},
])

# os dois links da entrega ficam no slide final, que e o que permanece na tela
# durante as perguntas
tb(s, M, 6.5, W, 0.55, [
    {"runs": [
        {"t": "Código  ", "sz": 10.5, "c": MUTED_DK},
        {"t": "github.com/GabrielFloriano03/Trabalho_SML_Profundo2",
         "sz": 10.5, "c": AMBER_LT, "font": MONO},
        {"t": "     Checkpoints  ", "sz": 10.5, "c": MUTED_DK},
        {"t": "huggingface.co/Floriano03/Trabalho_SML_Profundo2",
         "sz": 10.5, "c": AMBER_LT, "font": MONO},
    ]},
    {"t": "Gabriel Floriano  ·  Aprendizado Profundo 2  ·  PUCRS",
     "sz": 10, "c": MUTED_DK, "space_before": 4},
])
s.notes_slide.notes_text_frame.text = (
    "Fecho com a limitacao, nao com o resultado. O numero que resume o projeto nao e uma "
    "acuracia: e o 3,3% de ganho nos ultimos 200 M tokens, que diz com clareza que o "
    "proximo passo e aumentar o modelo, nao treinar mais."
)

prs.core_properties.title = "Small Language Model treinado do zero"
prs.core_properties.author = "Gabriel Floriano"
prs.core_properties.subject = "Aprendizado Profundo 2 - Projeto 2 - PUCRS"
prs.save(OUT)
print(f"ok: {OUT}  ({len(prs.slides._sldIdLst)} slides)")
