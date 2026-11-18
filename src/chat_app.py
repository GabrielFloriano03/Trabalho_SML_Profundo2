"""Portal de chat para demonstracao ao vivo do SLM (Streamlit).

Carrega o checkpoint final do SFT, mantem o historico da conversa e expoe os
parametros de geracao (temperatura, top-k, maximo de tokens). A resposta aparece
token a token, consumindo `model.generate_iter`.

A lista de checkpoints tem uma entrada por etapa do treino (pre-treino,
mid-training e SFT final), para comparar as tres ao vivo com a mesma pergunta.

Uso:
    streamlit run src/chat_app.py

O tema vem de .streamlit/config.toml; o resto do visual e o CSS em ESTILO.
"""

from __future__ import annotations

import html
import os
import sys
import time

import streamlit as st
import streamlit.components.v1 as components
import tiktoken
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat_format import build_prompt          # noqa: E402
from train_utils import load_model_from_checkpoint  # noqa: E402

ENC = tiktoken.get_encoding("gpt2")

# o modelo pode comecar um novo turno sozinho em vez de parar; cortamos ali
PARADAS = ("\nUser:", "User:", "\nAssistant:")
RETER = max(len(s) for s in PARADAS)

# perguntas ja testadas, para a demo ao vivo nao depender de improviso.
# as duas primeiras o modelo responde bem; a terceira e a quarta mostram o
# limite real dele (repeticao e aritmetica), que e parte do que temos a dizer.
SUGESTOES = [
    "What is the capital of France?",
    "Give me three tips to study more effectively.",
    "Explain photosynthesis in simple terms.",
    "If I have 3 apples and buy 7 more, how many do I have?",
]

# Uma entrada por etapa do projeto, e so. A pasta checkpoints/ tambem tem os
# _best/_last (que carregam estado do AdamW) e o sft_final_v1 (anterior a
# correcao do GSM8K); eles multiplicariam as opcoes da tela sem acrescentar
# nada a explicacao, entao ficam de fora da demo.
# O SFT final vem primeiro porque e o que o enunciado pede como padrao.
ETAPAS = [
    ("sft_final.pt", "SFT final"),
    ("midtrain_final.pt", "Mid-training"),
    ("pretrain_final.pt", "Pré-treino"),
]

st.set_page_config(page_title="SLM Chat — Projeto 2", page_icon="◆",
                   layout="centered", initial_sidebar_state="expanded")

ESTILO = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

:root {
  --bg: #0E1013;
  --surface: #171A1F;
  --surface-2: #1F242B;
  --border: #2A3038;
  --text: #E7E9EC;
  --muted: #8C949F;
  --accent: #E08A2E;
  --accent-dim: rgba(224, 138, 46, .12);
  --ui: Inter, -apple-system, "Segoe UI", Roboto, system-ui, sans-serif;
}

html, body, [class*="st-"], button, input, textarea { font-family: var(--ui); }
/* os icones do Streamlit sao ligaduras de uma fonte propria: a regra acima
   pega a classe deles e faria aparecer o nome do icone como texto */
[data-testid="stIconMaterial"] {
  font-family: "Material Symbols Rounded" !important;
}

/* cromo do Streamlit: deixa so o essencial, para a tela ficar limpa no projetor */
[data-testid="stToolbar"], footer { display: none !important; }
[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stDecoration"] { display: none; }

[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stMainBlockContainer"], .block-container {
  padding-top: 2.2rem; padding-bottom: 7rem; max-width: 790px;
}

/* ---------- cabecalho ---------- */
.topo { margin-bottom: 1.6rem; }
.topo h1 {
  font-size: 1.42rem; font-weight: 600; letter-spacing: -.015em;
  color: var(--text); margin: 0 0 .3rem 0; line-height: 1.2;
}
.topo h1 .dot { color: var(--accent); }
.topo p { color: var(--muted); font-size: .845rem; margin: 0; line-height: 1.5; }
.selos { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .85rem; }
.selo {
  font-size: .715rem; color: var(--muted); background: var(--surface);
  border: 1px solid var(--border); border-radius: 999px; padding: .2rem .62rem;
  white-space: nowrap;
}
.selo b { color: var(--text); font-weight: 500; }

/* ---------- mensagens ---------- */
.msg { display: flex; gap: .7rem; margin: 1.15rem 0; align-items: flex-start;
       animation: entra .22s ease-out; }
@keyframes entra { from { opacity: 0; transform: translateY(5px); } }

.marca {
  flex: 0 0 26px; width: 26px; height: 26px; border-radius: 50%;
  display: grid; place-items: center; font-size: .62rem; font-weight: 600;
  margin-top: .12rem; user-select: none;
}
.msg.assistant .marca { background: var(--accent-dim); color: var(--accent);
                        border: 1px solid rgba(224,138,46,.3); }
/* a mensagem do usuario nao leva marca: a bolha a direita ja diz de quem e */
.msg.user .marca { display: none; }

.bolha {
  font-size: .92rem; line-height: 1.68; color: var(--text);
  word-wrap: break-word; overflow-wrap: anywhere; max-width: 100%;
}
/* o modelo foi treinado em SmolTalk e escreve em markdown (listas, negrito), e
   o Streamlit renderiza isso; sem apertar os espacos, cada item de lista vira
   um paragrafo solto e a resposta fica cheia de buracos */
.bolha p { margin: 0 0 .62rem 0; }
.bolha > :last-child, .bolha p:last-child { margin-bottom: 0; }
.bolha ol, .bolha ul { margin: .45rem 0; padding-left: 1.35rem; }
.bolha li { margin: .2rem 0; }
.bolha li p { margin: 0; }
.bolha strong { color: var(--text); font-weight: 600; }
.bolha h1, .bolha h2, .bolha h3 {
  font-size: .95rem; font-weight: 600; margin: .7rem 0 .3rem 0; color: var(--text);
}
.bolha code {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 4px; padding: .05rem .28rem; font-size: .83em;
}
.bolha pre { background: var(--surface); border: 1px solid var(--border);
             border-radius: 8px; padding: .6rem .75rem; overflow-x: auto; }
.msg.user { flex-direction: row-reverse; }
.msg.user .bolha {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 14px 14px 4px 14px; padding: .6rem .85rem; max-width: 78%;
}
.msg.assistant .bolha { padding-top: .1rem; }
.msg.assistant.vazio .bolha { color: var(--muted); font-style: italic; }

.cursor {
  display: inline-block; width: 7px; height: 1.02em; background: var(--accent);
  vertical-align: text-bottom; margin-left: 2px; border-radius: 1px;
  animation: pisca 1s steps(2, start) infinite;
}
@keyframes pisca { 50% { opacity: 0; } }

.medida {
  font-size: .715rem; color: var(--muted); margin: -.55rem 0 1.3rem 2.35rem;
  font-variant-numeric: tabular-nums;
}

/* o iframe do helper de rolagem nao deve ocupar espaco entre as mensagens */
[data-testid="stElementContainer"]:has(iframe) {
  height: 0 !important; min-height: 0 !important; margin: 0 !important;
  overflow: hidden;
}
[data-testid="stElementContainer"] iframe { height: 0 !important; border: 0; }

/* ---------- estado vazio ---------- */
.hero {
  border: 1px solid var(--border); background: var(--surface);
  border-radius: 14px; padding: 1.15rem 1.25rem; margin: .4rem 0 1.05rem 0;
}
.hero h2 { font-size: .93rem; font-weight: 600; color: var(--text);
           margin: 0 0 .4rem 0; }
.hero p { font-size: .825rem; color: var(--muted); margin: 0; line-height: 1.6; }
.hero p + p { margin-top: .5rem; }
.hero code {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 4px; padding: .05rem .3rem; font-size: .78rem; color: var(--text);
}
.rotulo-sug {
  font-size: .715rem; text-transform: uppercase; letter-spacing: .07em;
  color: var(--muted); margin: 0 0 .55rem .1rem;
}

/* sugestoes: botoes do Streamlit com cara de chip */
[data-testid="stMainBlockContainer"] .stButton > button {
  background: var(--surface); color: var(--muted);
  border: 1px solid var(--border); border-radius: 10px;
  padding: .62rem .8rem; font-size: .82rem; font-weight: 400;
  line-height: 1.4; min-height: 0; transition: all .15s;
  /* o botao do Streamlit centraliza e corta o texto; aqui ele precisa
     alinhar a esquerda e quebrar linha, como um chip de sugestao */
  justify-content: flex-start; text-align: left; white-space: normal;
}
[data-testid="stMainBlockContainer"] .stButton > button:hover {
  border-color: var(--accent); color: var(--text); background: var(--surface-2);
}
/* o rotulo vive em button > div(flex) > span(flex) > div > p, e o span encolhe
   ate o tamanho do texto; sem esticar os dois, o chip fica centralizado */
[data-testid="stMainBlockContainer"] .stButton > button > div {
  width: 100%; justify-content: flex-start;
}
[data-testid="stMainBlockContainer"] .stButton > button > div > span {
  flex: 1 1 auto; width: 100%;
}
/* o <p> do rotulo e inline: quem alinha o texto e o bloco que o contem, e esse
   bloco encolhe ate a largura do texto se nao for esticado */
[data-testid="stMainBlockContainer"] .stButton > button div {
  text-align: left; width: 100%;
}
[data-testid="stMainBlockContainer"] .stButton > button span {
  justify-content: flex-start;
}
[data-testid="stMainBlockContainer"] .stButton > button p {
  font-size: .82rem; white-space: normal;
}

/* ---------- caixa de entrada ---------- */
[data-testid="stChatInput"] {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 13px;
}
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); }
[data-testid="stChatInput"] textarea { font-size: .9rem; color: var(--text); }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted); }
[data-testid="stBottomBlockContainer"] {
  background: var(--bg); max-width: 790px; padding-bottom: 1.1rem;
}

/* ---------- barra lateral ---------- */
[data-testid="stSidebar"] {
  background: var(--surface); border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 2.2rem; }
.sb-titulo {
  font-size: .7rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .09em; color: var(--muted);
  margin: 1.45rem 0 .6rem 0; padding-bottom: .4rem;
  border-bottom: 1px solid var(--border);
}
.sb-titulo:first-of-type { margin-top: 0; }
[data-testid="stSidebar"] label p { font-size: .8rem !important; color: var(--text); }
[data-testid="stSidebar"] .stButton > button {
  background: transparent; color: var(--muted); border: 1px solid var(--border);
  border-radius: 9px; font-size: .8rem; padding: .42rem;
}
[data-testid="stSidebar"] .stButton > button:hover {
  border-color: var(--accent); color: var(--accent);
}
.ficha { font-size: .745rem; color: var(--muted); line-height: 1.85;
         margin-bottom: .35rem; }
.ficha span { color: var(--text); float: right; font-variant-numeric: tabular-nums; }
.aviso {
  font-size: .745rem; color: var(--muted); line-height: 1.55;
  background: var(--accent-dim); border-left: 2px solid var(--accent);
  border-radius: 0 6px 6px 0; padding: .55rem .7rem; margin-top: .3rem;
}
</style>
"""
st.markdown(ESTILO, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
# max_entries=2 porque trocar de checkpoint na demo nao pode acumular modelos
# na VRAM: os _best/_last carregam estado do AdamW e sao 3x maiores
@st.cache_resource(show_spinner="Carregando o modelo...", max_entries=2)
def carregar(caminho: str, device: str):
    model, _ = load_model_from_checkpoint(caminho, device)
    model.eval()
    return model


def checkpoints() -> list[str]:
    """Os tres checkpoints de etapa, na ordem de ETAPAS."""
    caminhos = [os.path.join("checkpoints", nome) for nome, _ in ETAPAS]
    return [c for c in caminhos if os.path.exists(c)]


def rotulo(caminho: str) -> str:
    return dict(ETAPAS).get(os.path.basename(caminho), os.path.basename(caminho))


def recortar(texto: str) -> tuple[str, bool]:
    """Parte do texto que ja pode ir para a tela, e se devemos parar.

    Enquanto nao aparece sequencia de parada, seguramos os ultimos caracteres:
    eles podem ser o inicio de um "\\nUser:" que vamos querer cortar, e nao da
    para apagar o que ja foi exibido.
    """
    for p in PARADAS:
        i = texto.find(p)
        if i != -1:
            return texto[:i], True
    return (texto[:-RETER] if len(texto) > RETER else ""), False


def responder(model, device, historico, temperatura, top_k, max_novos, turnos):
    """Gera a resposta e devolve (texto, n_tokens, segundos), token a token.

    Implementado como gerador para o chamador poder desenhar a bolha enquanto a
    resposta sai; o valor final vem em `.value` no fim da iteracao.
    """
    # so os ultimos turnos entram no prompt: o contexto e curto (1024 tokens)
    recente = historico[-(2 * turnos + 1):] if turnos else historico[-1:]
    prompt = build_prompt(recente)

    ids = [ENC.eot_token] + ENC.encode_ordinary(prompt)
    ids = ids[-(model.cfg.max_seq_len - max_novos - 1):]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    saida: list[int] = []
    exibido = 0
    t0 = time.perf_counter()

    for token in model.generate_iter(x, max_novos, temperatura,
                                     top_k if top_k > 0 else None,
                                     eos_id=ENC.eot_token):
        tid = int(token[0, 0])
        if tid == ENC.eot_token:
            break
        saida.append(tid)
        # decodificamos a lista inteira a cada passo em vez de token a token:
        # um caractere pode ocupar varios tokens BPE, e decodificar em pedacos
        # produz caracteres quebrados no meio da palavra
        visivel, parar = recortar(ENC.decode(saida))
        if len(visivel) > exibido:
            yield visivel[exibido:]
            exibido = len(visivel)
        if parar:
            break

    texto = ENC.decode(saida)
    for p in PARADAS:
        i = texto.find(p)
        if i != -1:
            texto = texto[:i]
            break
    texto = texto.strip()
    if len(texto) > exibido:
        yield texto[exibido:]

    return texto, len(saida), time.perf_counter() - t0


def rolar(seguir: bool = False):
    """Leva a conversa para o fim da pagina.

    O auto-scroll do Streamlit so vale para st.chat_message, e aqui as bolhas
    sao HTML nosso; sem isto a resposta nasce abaixo da dobra. Com seguir=True
    a pagina acompanha a resposta enquanto ela sai, ate o usuario rolar para
    cima (ai paramos, para nao brigar com quem quer reler o historico).
    """
    components.html(
        f"""<script>
        const d = window.parent.document;
        const m = d.querySelector('[data-testid="stAppScrollToBottomContainer"]');
        if (m) {{
          clearInterval(window.parent.__slmRolagem);
          m.scrollTop = m.scrollHeight;
          if ({str(seguir).lower()}) {{
            let ativo = true;
            m.addEventListener('wheel', () => ativo = false, {{once: true, passive: true}});
            window.parent.__slmRolagem = setInterval(() => {{
              if (ativo) m.scrollTop = m.scrollHeight;
            }}, 150);
          }}
        }}
        </script>""", height=0)


def bolha(papel: str, texto: str, *, cursor: bool = False) -> str:
    marca = "◆" if papel == "assistant" else "você"
    corpo = html.escape(texto)
    if cursor:
        corpo += '<span class="cursor"></span>'
    vazio = " vazio" if not texto and not cursor else ""
    return (f'<div class="msg {papel}{vazio}">'
            f'<div class="marca">{marca}</div>'
            f'<div class="bolha">{corpo or "(sem resposta)"}</div></div>')


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown('<div class="sb-titulo">Modelo</div>', unsafe_allow_html=True)
    disponiveis = checkpoints()
    if not disponiveis:
        st.error("Nenhum checkpoint em checkpoints/")
        st.stop()
    ckpt = st.selectbox("Checkpoint", disponiveis, index=0,
                        format_func=rotulo, label_visibility="collapsed")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = carregar(ckpt, device)
    if device == "cuda":
        # "NVIDIA GeForce RTX 3050 6GB Laptop GPU" nao cabe na ficha
        gpu = (torch.cuda.get_device_name(0)
               .replace("NVIDIA ", "").replace("GeForce ", "")
               .replace(" Laptop GPU", "").replace(" GPU", ""))
    else:
        gpu = "CPU"
    st.markdown(
        f'<div class="ficha">'
        f'Parâmetros<span>{model.num_params()/1e6:.2f} M</span><br>'
        f'Contexto<span>{model.cfg.max_seq_len} tokens</span><br>'
        f'Camadas<span>{model.cfg.n_layer} × {model.cfg.d_model}</span><br>'
        f'Dispositivo<span>{gpu}</span>'
        f'</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-titulo">Geração</div>', unsafe_allow_html=True)
    temperatura = st.slider("Temperatura", 0.0, 1.5, 0.8, 0.05,
                            help="0 = greedy (determinístico). Acima de ~1,1 "
                                 "o texto desanda.")
    top_k = st.slider("Top-k", 0, 200, 50, 5, help="0 desativa o top-k")
    max_novos = st.slider("Máximo de tokens novos", 16, 512, 160, 16)
    turnos = st.slider("Turnos de histórico no prompt", 0, 8, 3, 1,
                       help="Quantos pares pergunta/resposta anteriores entram "
                            "no prompt. O contexto é de 1024 tokens.")

    st.markdown('<div class="sb-titulo">Conversa</div>', unsafe_allow_html=True)
    if st.button("Limpar conversa", use_container_width=True):
        st.session_state.mensagens = []
        st.rerun()
    st.markdown(
        '<div class="aviso">Treinado só em inglês (FineWeb-Edu + SmolTalk). '
        'Pergunte em inglês — em português ele não responde.</div>',
        unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Conversa
# --------------------------------------------------------------------------- #
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

st.markdown(
    '<div class="topo">'
    '<h1>SLM Chat <span class="dot">◆</span></h1>'
    '<p>Small Language Model treinado do zero — Projeto 2, Aprendizado Profundo 2</p>'
    '<div class="selos">'
    '<div class="selo"><b>55,26 M</b> parâmetros</div>'
    '<div class="selo"><b>1,10 B</b> tokens de pré-treino</div>'
    '<div class="selo"><b>24 h</b> de GPU</div>'
    '<div class="selo"><b>6 GB</b> de VRAM</div>'
    '</div></div>', unsafe_allow_html=True)

# vem de um clique nas sugestoes, resolvido antes do estado vazio para o cartao
# de boas-vindas nao reaparecer junto com a primeira resposta
pendente = st.session_state.pop("pendente", None)

for m in st.session_state.mensagens:
    st.markdown(bolha(m["papel"], m["texto"]), unsafe_allow_html=True)
    if m.get("medida"):
        st.markdown(f'<div class="medida">{m["medida"]}</div>',
                    unsafe_allow_html=True)

if not st.session_state.mensagens and not pendente:
    st.markdown(
        '<div class="hero">'
        '<h2>O que esperar</h2>'
        '<p>O modelo tem 55 M de parâmetros — cerca de 2.500× menos que o '
        'GPT-4. Ele acerta o formato de conversa e fatos simples; erra '
        'aritmética e repete frases em respostas longas.</p>'
        '<p>A mesma pergunta pode ser feita às três etapas do treino: troque '
        'entre <code>Pré-treino</code>, <code>Mid-training</code> e '
        '<code>SFT final</code> na barra lateral.</p>'
        '</div>', unsafe_allow_html=True)

    st.markdown('<div class="rotulo-sug">Perguntas de teste</div>',
                unsafe_allow_html=True)
    colunas = st.columns(2)
    for i, s in enumerate(SUGESTOES):
        if colunas[i % 2].button(s, key=f"sug{i}", use_container_width=True):
            st.session_state.pendente = s
            st.rerun()

digitado = st.chat_input("Escreva sua mensagem em inglês...")
if digitado:
    # o chat_input so devolve o texto depois que o resto da pagina ja desenhou,
    # entao o cartao de boas-vindas apareceria acima da primeira pergunta;
    # guardar e refazer a pagina deixa os dois caminhos de entrada iguais
    st.session_state.pendente = digitado
    st.rerun()

if pendente:
    st.session_state.mensagens.append({"papel": "user", "texto": pendente})
    st.markdown(bolha("user", pendente), unsafe_allow_html=True)

    historico = [{"role": "user" if m["papel"] == "user" else "assistant",
                  "content": m["texto"]} for m in st.session_state.mensagens]

    vaga = st.empty()
    vaga.markdown(bolha("assistant", "", cursor=True), unsafe_allow_html=True)
    rolar(seguir=True)

    fluxo = responder(model, device, historico, temperatura, top_k,
                      max_novos, turnos)
    texto = ""
    ultimo = 0.0
    while True:
        try:
            texto += next(fluxo)
        except StopIteration as fim:
            resposta, n_tokens, segundos = fim.value
            break
        # o gargalo e o ida-e-volta do websocket, nao o modelo: redesenhar a
        # bolha a cada token deixa a geracao mais lenta que ela e
        agora = time.perf_counter()
        if agora - ultimo > 0.05:
            vaga.markdown(bolha("assistant", texto, cursor=True),
                          unsafe_allow_html=True)
            ultimo = agora

    vaga.markdown(bolha("assistant", resposta), unsafe_allow_html=True)
    medida = (f"{n_tokens} tokens · {segundos:.1f} s · "
              f"{n_tokens / segundos:.0f} tokens/s" if segundos > 0 else "")
    if medida:
        st.markdown(f'<div class="medida">{medida}</div>', unsafe_allow_html=True)

    st.session_state.mensagens.append({"papel": "assistant", "texto": resposta,
                                       "medida": medida})
    rolar()
