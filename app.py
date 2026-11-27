"""WattCast - Prototipo de pronóstico del consumo mensual de electrodomésticos.

Datos: Candanedo, Feldheim y Deramaix (2017), "Appliances Energy Prediction".
Modelo: Random Forest (100 árboles), el mejor modelo según la comparación hecha en Orange.
El modelo predice el consumo de cada intervalo de 10 minutos; la app suma las predicciones de las
24 horas del día y las multiplica por 30 días para estimar el consumo mensual.
Código generado con apoyo de inteligencia artificial (Claude, Anthropic) y revisado por el equipo.

Ejecutar:  streamlit run app.py   (el archivo energydata_complete.csv debe estar en la misma carpeta)
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA = Path(__file__).parent / "energydata_complete.csv"
ROOMS = [1, 2, 3, 4, 5, 7, 8, 9]          # ambientes interiores (T6/RH_6 están fuera de la casa)
LOW, HIGH = 400, 625                      # kWh/mes: mediana y ~percentil 90 del consumo diario x 30 días

# Valores iniciales y escenarios rápidos
DEFAULTS = {"hora": 18, "lights": 10, "t_in": 20.0, "rh_in": 42, "t_out": 7.0, "rh_out": 84, "wind": 3.5}
PRESETS = {
    "🌙 Madrugada": {"hora": 3, "lights": 0, "t_in": 19.5, "rh_in": 41, "t_out": 5.0, "rh_out": 90, "wind": 3.0},
    "☀️ Mañana": {"hora": 9, "lights": 20, "t_in": 20.5, "rh_in": 43, "t_out": 8.0, "rh_out": 80, "wind": 4.0},
    "🔥 Hora punta": DEFAULTS,
    "❄️ Noche fría": {"hora": 21, "lights": 30, "t_in": 18.0, "rh_in": 45, "t_out": 0.0, "rh_out": 92, "wind": 6.0},
}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA, parse_dates=["date"])
    df["hora"] = df["date"].dt.hour
    return df


@st.cache_resource
def train(df):
    X = df.drop(columns=["date", "Appliances", "rv1", "rv2"])
    y = df["Appliances"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.75, random_state=0)
    model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    metrics = {"RMSE": mean_squared_error(yte, p) ** 0.5, "MAE": mean_absolute_error(yte, p),
               "R2": r2_score(yte, p), "n_test": len(yte)}
    return model, list(X.columns), metrics


@st.cache_data
def monthly_reference(df):
    """Consumo mensual equivalente (kWh) de cada día completo del conjunto de datos."""
    day = df.groupby(df["date"].dt.date)["Appliances"].agg(["sum", "count"])
    day = day[day["count"] == 144]
    return (day["sum"] / 1000 * 30).to_numpy()


def build_row(df, hora, lights, t_in, rh_in, t_out, rh_out, wind, press, vis, dew):
    """Arma las 26 variables del modelo a partir de los datos que ingresa el usuario."""
    med = df.median(numeric_only=True)
    t_shift = t_in - np.mean([med[f"T{i}"] for i in ROOMS])
    rh_shift = rh_in - np.mean([med[f"RH_{i}"] for i in ROOMS])
    row = {"lights": lights, "T_out": t_out, "RH_out": rh_out, "Windspeed": wind,
           "Press_mm_hg": press, "Visibility": vis, "Tdewpoint": dew, "hora": hora,
           "T6": t_out, "RH_6": rh_out}
    for i in ROOMS:
        row[f"T{i}"] = med[f"T{i}"] + t_shift
        row[f"RH_{i}"] = med[f"RH_{i}"] + rh_shift
    return row


def nivel(kwh_mes):
    if kwh_mes <= LOW:
        return "BAJO", "#22c55e"
    if kwh_mes <= HIGH:
        return "MEDIO", "#f59e0b"
    return "ALTO", "#ef4444"


def aplicar_escenario(nombre):
    for k, v in PRESETS[nombre].items():
        st.session_state[k] = v


st.set_page_config(page_title="WattCast", page_icon="⚡", layout="centered")

# ---------------------------------------------------------------- estilos generales
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap');
html, body, .stMarkdown, button, input, label, p {font-family:'Outfit',system-ui,sans-serif}
.stApp{background:radial-gradient(1200px 500px at 50% -120px,#fff1cc 0%,#ffffff 62%)}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:2.2rem;max-width:800px}
.wc-hero{position:relative;overflow:hidden;border-radius:22px;padding:30px 24px 24px;text-align:center;color:#fff;
  background:linear-gradient(135deg,#0b1020 0%,#16213e 55%,#3b1d4a 100%);box-shadow:0 18px 40px -18px #16213e}
.wc-hero:before{content:"";position:absolute;inset:-60%;background:conic-gradient(from 0deg,transparent 0 70%,#f9c80e40 80%,transparent 90%);
  animation:wcSpin 8s linear infinite}
.wc-hero>*{position:relative}
.wc-logo{font-size:3.6rem;font-weight:800;letter-spacing:-.03em;line-height:1.05;
  background:linear-gradient(90deg,#ffe066,#f9a825 35%,#ff6b6b 70%,#ffe066);background-size:200% auto;
  -webkit-background-clip:text;background-clip:text;color:transparent!important;-webkit-text-fill-color:transparent;
  animation:wcShine 4s linear infinite}
.wc-bolt{display:inline-block;font-size:3rem;margin-right:.12em;animation:wcZap 2.6s ease-in-out infinite}
.wc-sub{color:#c9cfe0;margin:10px auto 14px;max-width:520px;font-size:1.02rem}
.wc-chips{display:flex;flex-wrap:wrap;justify-content:center;gap:8px}
.wc-chips span{font-size:.82rem;padding:5px 12px;border-radius:999px;background:#ffffff14;border:1px solid #ffffff2a;color:#e9ecf5}
.wc-h2{font-weight:800;font-size:1.45rem;margin:26px 0 8px;color:#16213e}
.wc-h2 small{font-weight:400;color:#6b7280;font-size:.92rem;margin-left:6px}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px!important;background:#ffffffd9;box-shadow:0 8px 26px -16px #16213e88}
.wc-card-t{font-weight:800;color:#16213e;margin-bottom:2px;font-size:1.05rem}
div.stButton>button{border-radius:12px;font-weight:600;transition:transform .15s,box-shadow .15s}
div.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 18px -10px #f9a825;border-color:#f9a825}
div.stButton>button[kind="primary"]{background:linear-gradient(90deg,#f9a825,#ff6b6b);border:0;color:#fff;
  padding:.75rem 0;box-shadow:0 10px 24px -12px #ff6b6b;animation:wcPulse 2.4s ease-in-out infinite}
div.stButton>button[kind="primary"] p{font-size:1.15rem;font-weight:800}
.wc-foot{color:#9aa1b2;font-size:.82rem;text-align:center;margin-top:18px}
@keyframes wcSpin{to{transform:rotate(360deg)}}
@keyframes wcShine{to{background-position:200% center}}
@keyframes wcZap{0%,86%,100%{transform:none;filter:none}88%{transform:rotate(-12deg) scale(1.15);filter:drop-shadow(0 0 10px #ffe066)}92%{transform:rotate(8deg)}}
@keyframes wcPulse{0%,100%{box-shadow:0 10px 24px -12px #ff6b6b}50%{box-shadow:0 12px 32px -6px #ff6b6b}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- animación de entrada (solo CSS)
if "intro_vista" not in st.session_state:          # aparece solo al abrir la app
    st.session_state.intro_vista = True
    rnd = random.Random(7)
    chispas = "".join(
        f"<i style='--a:{rnd.uniform(0, 360):.0f}deg;--d:{rnd.uniform(140, 420):.0f}px;"
        f"--t:{rnd.uniform(.7, 1.3):.2f}s;--w:{rnd.uniform(2, 5):.1f}px;"
        f"--c:{rnd.choice(['#ffe066', '#f9a825', '#ff6b6b', '#7dd3fc'])}'></i>"
        for _ in range(42))
    letras = "".join(
        f"<b class='{'y' if i >= 4 else ''}' style='animation-delay:{1.55 + i * 0.07:.2f}s'>{c}</b>"
        for i, c in enumerate("WattCast"))
    circuitos = "".join(
        f"<path d='{d}' style='animation-delay:{k * 0.12:.2f}s'/>" for k, d in enumerate([
            "M-50 120 H260 V300 H430", "M1050 90 H760 V260 H570", "M-50 640 H220 V420 H430",
            "M1050 650 H800 V440 H570", "M500 -40 V190", "M500 760 V420"]))
    st.markdown(f"""
<style>
.wc-intro{{position:fixed;inset:0;z-index:999999;pointer-events:none;overflow:hidden}}
.wc-half{{position:absolute;left:0;right:0;height:50.5%;background:#070a12;animation:wcSplit .8s cubic-bezier(.7,0,.3,1) 3.7s forwards}}
.wc-half.t{{top:0;--dir:-1}} .wc-half.b{{bottom:0;--dir:1}}
.wc-stage{{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;
  animation:wcStageOut .5s ease-in 3.4s forwards}}
.wc-grid{{position:absolute;left:-50%;right:-50%;top:45%;bottom:-60%;
  background-image:linear-gradient(#26345e 1px,transparent 1px),linear-gradient(90deg,#26345e 1px,transparent 1px);
  background-size:46px 46px;transform:perspective(500px) rotateX(62deg);opacity:0;
  animation:wcGridIn 1s ease .1s forwards,wcGridMove 1.2s linear infinite}}
.wc-circ{{position:absolute;inset:0;width:100%;height:100%}}
.wc-circ path{{fill:none;stroke:#f9c80e;stroke-width:2.4;stroke-dasharray:900;stroke-dashoffset:900;filter:drop-shadow(0 0 6px #f9c80e);
  animation:wcDraw 1.1s cubic-bezier(.5,0,.2,1) forwards}}
.wc-core{{position:relative;width:170px;height:170px;display:flex;align-items:center;justify-content:center}}
.wc-core svg.bolt{{width:130px;height:130px;opacity:0;transform:scale(.3) rotate(-20deg);
  animation:wcBoltIn .45s cubic-bezier(.2,1.6,.4,1) 1.05s forwards,wcBoltGlow 1.2s ease-in-out 1.5s infinite alternate}}
.wc-core svg.bolt path{{fill:#ffe066;stroke:#fff6c2;stroke-width:1.5;stroke-linejoin:round}}
.wc-ring{{position:absolute;inset:0;border-radius:50%;border:3px solid #f9c80e;opacity:0;animation:wcShock 1.1s ease-out forwards}}
.wc-ring.r1{{animation-delay:1.15s}} .wc-ring.r2{{animation-delay:1.35s;border-color:#ff6b6b}} .wc-ring.r3{{animation-delay:1.6s;border-color:#7dd3fc}}
.wc-sparks i{{position:absolute;left:50%;top:50%;width:var(--w);height:var(--w);border-radius:50%;background:var(--c);
  box-shadow:0 0 10px var(--c);opacity:0;animation:wcSpark var(--t) ease-out 1.15s forwards}}
.wc-flash{{position:absolute;inset:0;background:#fff;opacity:0;animation:wcFlash .4s ease-out 1.05s}}
.wc-name{{display:flex;gap:1px;margin-top:18px;font:800 4.2rem/1 'Outfit',sans-serif;letter-spacing:-.03em;color:#fff}}
.wc-name b{{opacity:0;transform:translateY(40px) scale(.6) rotate(-8deg);filter:blur(8px);
  animation:wcLetter .55s cubic-bezier(.2,1.4,.4,1) forwards}}
.wc-name b.y{{color:#ffe066;text-shadow:0 0 18px #f9a82599}}
.wc-tag{{margin-top:12px;color:#aab3cc;font:400 1.05rem 'Outfit',sans-serif;letter-spacing:.02em;opacity:0;animation:wcFade .6s ease 2.3s forwards}}
.wc-load{{margin-top:22px;width:260px;opacity:0;animation:wcFade .4s ease .4s forwards}}
.wc-bar{{height:6px;border-radius:4px;background:#1b2440;overflow:hidden}}
.wc-bar i{{display:block;height:100%;width:0;background:linear-gradient(90deg,#22c55e,#f9c80e,#ff6b6b);box-shadow:0 0 12px #f9c80e;
  animation:wcCharge 2.9s cubic-bezier(.6,0,.2,1) .4s forwards}}
@property --pct{{syntax:'<integer>';initial-value:0;inherits:false}}
.wc-pct{{margin-top:8px;color:#f9c80e;font:600 .95rem 'Outfit',sans-serif;text-align:right;counter-reset:p var(--pct);
  animation:wcCount 2.9s cubic-bezier(.6,0,.2,1) .4s forwards}}
.wc-pct:after{{content:"Cargando energía… " counter(p) "%"}}
@keyframes wcGridIn{{to{{opacity:.6}}}}
@keyframes wcGridMove{{to{{background-position:0 46px}}}}
@keyframes wcDraw{{to{{stroke-dashoffset:0}}}}
@keyframes wcBoltIn{{to{{opacity:1;transform:none}}}}
@keyframes wcBoltGlow{{from{{filter:drop-shadow(0 0 8px #f9c80e)}}to{{filter:drop-shadow(0 0 34px #ffe066)}}}}
@keyframes wcShock{{0%{{opacity:.9;transform:scale(.6)}}100%{{opacity:0;transform:scale(3.4)}}}}
@keyframes wcSpark{{0%{{opacity:1;transform:translate(-50%,-50%) rotate(var(--a)) translateX(0)}}
  100%{{opacity:0;transform:translate(-50%,-50%) rotate(var(--a)) translateX(var(--d))}}}}
@keyframes wcFlash{{0%{{opacity:.85}}100%{{opacity:0}}}}
@keyframes wcLetter{{to{{opacity:1;transform:none;filter:none}}}}
@keyframes wcFade{{to{{opacity:1}}}}
@keyframes wcCharge{{to{{width:100%}}}}
@keyframes wcCount{{to{{--pct:100}}}}
@keyframes wcStageOut{{to{{opacity:0;transform:scale(1.15);filter:blur(6px)}}}}
@keyframes wcSplit{{to{{transform:translateY(calc(var(--dir) * 105%))}}}}
@media (prefers-reduced-motion:reduce){{.wc-intro{{display:none}}}}
</style>
<div class="wc-intro" aria-hidden="true">
  <div class="wc-half t"></div><div class="wc-half b"></div>
  <div class="wc-stage">
    <div class="wc-grid"></div>
    <svg class="wc-circ" viewBox="0 0 1000 720" preserveAspectRatio="xMidYMid slice">{circuitos}</svg>
    <div class="wc-core">
      <span class="wc-ring r1"></span><span class="wc-ring r2"></span><span class="wc-ring r3"></span>
      <span class="wc-sparks">{chispas}</span>
      <svg class="bolt" viewBox="0 0 64 64"><path d="M36 3 L11 36 H29 L24 61 L53 25 H35 Z"/></svg>
    </div>
    <div class="wc-name">{letras}</div>
    <div class="wc-tag">Pronóstico inteligente del consumo de tu hogar</div>
    <div class="wc-load"><div class="wc-bar"><i></i></div><div class="wc-pct"></div></div>
    <div class="wc-flash"></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- datos y modelo
df = load_data()
with st.spinner("Preparando el modelo (solo la primera vez)..."):
    model, cols, metrics = train(df)
med = df.median(numeric_only=True)
ref_mes = monthly_reference(df)

st.markdown(f"""
<div class="wc-hero">
  <div><span class="wc-bolt">⚡</span><span class="wc-logo">WattCast</span></div>
  <div class="wc-sub">Pronóstico del consumo mensual de los electrodomésticos de una vivienda
  con Random Forest (datos de Candanedo et al., 2017).</div>
  <div class="wc-chips"><span>🌲 Random Forest · 100 árboles</span><span>🎯 R² = {metrics['R2']:.2f}</span>
  <span>📊 {len(df):,} registros</span></div>
</div>""".replace(",", " ", 1) if False else f"""
<div class="wc-hero">
  <div><span class="wc-bolt">⚡</span><span class="wc-logo">WattCast</span></div>
  <div class="wc-sub">Pronóstico del consumo mensual de los electrodomésticos de una vivienda
  con Random Forest (datos de Candanedo et al., 2017).</div>
  <div class="wc-chips"><span>🌲 Random Forest · 100 árboles</span><span>🎯 R² = {metrics['R2']:.2f}</span>
  <span>📊 {len(df):,} registros</span></div>
</div>""".replace(f"{len(df):,}", f"{len(df):,}".replace(",", " ")), unsafe_allow_html=True)

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ---------------------------------------------------------------- entradas
st.markdown("<div class='wc-h2'>1. Ingrese las condiciones <small>o elija un escenario rápido</small></div>",
            unsafe_allow_html=True)
bcols = st.columns(len(PRESETS))
for col, nombre in zip(bcols, PRESETS):
    col.button(nombre, on_click=aplicar_escenario, args=(nombre,), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("<div class='wc-card-t'>🏠 Dentro de la casa</div>", unsafe_allow_html=True)
        hora = st.slider("Hora del día", 0, 23, key="hora", format="%d h")
        lights = st.select_slider("Consumo de las luces (Wh)", options=list(range(0, 80, 10)), key="lights")
        t_in = st.slider("Temperatura interior promedio (°C)", 15.0, 26.0, step=0.5, key="t_in")
        rh_in = st.slider("Humedad interior promedio (%)", 28, 60, key="rh_in")
with c2:
    with st.container(border=True):
        st.markdown("<div class='wc-card-t'>🌤️ Clima exterior</div>", unsafe_allow_html=True)
        t_out = st.slider("Temperatura exterior (°C)", -5.0, 26.0, step=0.5, key="t_out")
        rh_out = st.slider("Humedad exterior (%)", 24, 100, key="rh_out")
        wind = st.slider("Velocidad del viento (m/s)", 0.0, 14.0, step=0.5, key="wind")
        with st.expander("Opciones avanzadas"):
            press = st.number_input("Presión atmosférica (mm Hg)", 729.0, 773.0, float(round(med["Press_mm_hg"], 1)))
            vis = st.number_input("Visibilidad (km)", 1.0, 66.0, float(med["Visibility"]))
            dew = st.number_input("Punto de rocío (°C)", -7.0, 16.0, float(round(med["Tdewpoint"], 1)))

if st.button("⚡ Predecir consumo", type="primary", use_container_width=True):
    st.session_state.mostrar = True

# ---------------------------------------------------------------- resultado
if st.session_state.get("mostrar"):
    rows = [build_row(df, h, lights, t_in, rh_in, t_out, rh_out, wind, press, vis, dew) for h in range(24)]
    wh10 = model.predict(pd.DataFrame(rows)[cols])          # Wh en 10 minutos, para cada hora
    wh_hora = wh10 * 6                                       # Wh en cada hora
    kwh_dia = wh_hora.sum() / 1000
    kwh_mes = kwh_dia * 30
    txt, color = nivel(kwh_mes)
    prom = float(ref_mes.mean())
    datos = {
        "mes": round(kwh_mes, 1), "dia": round(kwh_dia, 2), "nivel": txt, "color": color,
        "hora": int(hora), "wh10": round(float(wh10[hora]), 1), "perfil": [round(float(v), 1) for v in wh_hora],
        "low": LOW, "high": HIGH, "prom": round(prom, 1),
        "vs": round((kwh_mes - prom) / prom * 100), "pct": round(float((ref_mes < kwh_mes).mean() * 100)),
        "pico": int(np.argmax(wh_hora)), "valle": int(np.argmin(wh_hora)),
    }
    st.markdown("<div class='wc-h2'>2. Resultado</div>", unsafe_allow_html=True)
    components.html(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap');
*{box-sizing:border-box}body{margin:0;padding:2px 2px 10px;font-family:'Outfit',system-ui,sans-serif;color:#16213e;background:transparent}
.card{position:relative;overflow:hidden;border-radius:20px;padding:20px;background:linear-gradient(135deg,#0b1020,#16213e 60%,#2a1b3d);color:#fff;
  box-shadow:0 18px 36px -20px #16213e;animation:rise .6s cubic-bezier(.2,1,.3,1)}
.card:before{content:"";position:absolute;width:320px;height:320px;left:-80px;top:-120px;border-radius:50%;
  background:radial-gradient(circle,var(--c) 0%,transparent 65%);opacity:.28;animation:breathe 3s ease-in-out infinite}
.card>*{position:relative}
.lbl{font-size:.85rem;color:#aab3cc;letter-spacing:.04em;text-transform:uppercase;margin-bottom:4px}
.top{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:18px;align-items:center}
.gauge{text-align:center}
.gauge svg{width:100%;max-width:300px;overflow:visible}
.big{font-size:2.7rem;font-weight:800;line-height:1;margin-top:-64px}
.big small{font-size:1rem;font-weight:600;color:#aab3cc;margin-left:4px}
.pill{display:inline-block;margin-top:12px;padding:5px 16px;border-radius:999px;font-weight:800;letter-spacing:.05em;
  background:var(--c);color:#0b1020;box-shadow:0 0 22px var(--c);animation:pop .5s cubic-bezier(.2,1.6,.4,1) 1.4s both}
.kpis{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.k{background:#ffffff10;border:1px solid #ffffff1f;border-radius:14px;padding:12px;opacity:0;animation:rise .5s ease forwards}
.k b{display:block;font-size:1.3rem;font-weight:800}
.k span{font-size:.8rem;color:#aab3cc}
.note{font-size:.8rem;color:#aab3cc;margin-top:14px;border-top:1px solid #ffffff1a;padding-top:10px}
.chart{margin-top:16px;border-radius:20px;padding:18px 16px 10px;background:#fff;box-shadow:0 10px 30px -18px #16213e88;
  opacity:0;animation:rise .6s ease .3s forwards}
.chart h4{margin:0 0 4px;font-size:1.05rem}
.chart p{margin:0 0 6px;font-size:.84rem;color:#6b7280}
svg text{font-family:'Outfit',sans-serif}
.bar{transform-origin:bottom;transform-box:fill-box;transform:scaleY(0);animation:grow .7s cubic-bezier(.2,1,.3,1) forwards}
@keyframes grow{to{transform:scaleY(1)}}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes pop{from{opacity:0;transform:scale(.4)}to{opacity:1;transform:none}}
@keyframes breathe{50%{opacity:.45;transform:scale(1.1)}}
@media(max-width:560px){.top{grid-template-columns:1fr}.big{font-size:2.3rem}}
</style>
<div class="card" id="card">
  <div class="lbl">Consumo mensual estimado de los electrodomésticos</div>
  <div class="top">
    <div class="gauge">
      <svg viewBox="0 0 300 175" id="g"></svg>
      <div class="big"><span id="num">0</span><small>kWh/mes</small></div>
      <div class="pill" id="pill"></div>
    </div>
    <div class="kpis" id="kpis"></div>
  </div>
  <div class="note" id="note"></div>
</div>
<div class="chart">
  <h4>📊 Consumo estimado por hora con las mismas condiciones</h4>
  <p id="cp"></p>
  <svg id="ch" width="100%" height="230"></svg>
</div>
<script>
const D = __DATA__;
const f = (x, d = 0) => Number(x).toLocaleString('es-PE', {minimumFractionDigits: d, maximumFractionDigits: d});
document.getElementById('card').style.setProperty('--c', D.color);
const pill = document.getElementById('pill'); pill.style.setProperty('--c', D.color); pill.textContent = 'NIVEL ' + D.nivel;
// medidor semicircular 0-900 kWh/mes
const MAX = 900, cx = 150, cy = 150, R = 118, g = document.getElementById('g');
const pt = (v, r) => { const a = Math.PI * (1 - Math.min(v, MAX) / MAX); return [cx + r * Math.cos(a), cy - r * Math.sin(a)]; };
const arc = (a, b, col) => { const [x1, y1] = pt(a, R), [x2, y2] = pt(b, R);
  return `<path d="M${x1} ${y1} A${R} ${R} 0 0 1 ${x2} ${y2}" stroke="${col}" stroke-width="18" fill="none"/>`; };
let s = `<path d="M32 150 A118 118 0 0 1 268 150" stroke="#ffffff14" stroke-width="30" fill="none"/>`;
s += arc(0, D.low, '#22c55e') + arc(D.low, D.high, '#f59e0b') + arc(D.high, MAX, '#ef4444');
[0, 300, 600, 900].forEach(v => { const [x, y] = pt(v, R + 22); s += `<text x="${x}" y="${y + 4}" fill="#aab3cc" font-size="11" text-anchor="middle">${v}</text>`; });
s += `<g id="nd" style="transform-origin:150px 150px;transform:rotate(-90deg);transition:transform 1.6s cubic-bezier(.2,1.3,.3,1)">
  <line x1="150" y1="150" x2="150" y2="44" stroke="#fff" stroke-width="4" stroke-linecap="round"/>
  <circle cx="150" cy="150" r="10" fill="#fff"/><circle cx="150" cy="150" r="4" fill="${D.color}"/></g>`;
g.innerHTML = s;
setTimeout(() => { document.getElementById('nd').style.transform = `rotate(${-90 + 180 * Math.min(D.mes, MAX) / MAX}deg)`; }, 150);
// número que sube
const el = document.getElementById('num'), t0 = performance.now();
(function tick(t) { const p = Math.min((t - t0) / 1500, 1), e = 1 - Math.pow(1 - p, 3);
  el.textContent = f(D.mes * e); if (p < 1) requestAnimationFrame(tick); })(t0);
// tarjetas
const K = [['📅 ' + f(D.dia, 1) + ' kWh', 'consumo por día'], ['⏱️ ' + f(D.wh10) + ' Wh', `cada 10 min a las ${D.hora} h`],
  [(D.vs > 0 ? '📈 +' : '📉 ') + f(D.vs) + ' %', `frente al promedio de la casa (${f(D.prom)} kWh/mes)`],
  [`🔺 ${D.pico} h · 🔻 ${D.valle} h`, 'hora pico · hora valle']];
document.getElementById('kpis').innerHTML = K.map((k, i) => `<div class="k" style="animation-delay:${.4 + i * .12}s"><b>${k[0]}</b><span>${k[1]}</span></div>`).join('');
document.getElementById('note').textContent = `Este pronóstico supera al ${D.pct} % de los días registrados. Bajo: hasta ${D.low} kWh/mes · Medio: ${D.low}–${D.high} · Alto: más de ${D.high}. Se calcula con 30 días iguales y las condiciones ingresadas durante las 24 horas.`;
document.getElementById('cp').textContent = `La barra naranja es la hora elegida (${D.hora} h). Pico a las ${D.pico} h con ${f(D.perfil[D.pico])} Wh.`;
// gráfico por hora
const ch = document.getElementById('ch');
function draw() {
  const W = ch.clientWidth || 600, H = 230, L = 44, Rr = 6, T = 18, B = 26, mx = Math.ceil(Math.max(...D.perfil) / 200) * 200;
  const bw = (W - L - Rr) / 24, y = v => T + (H - T - B) * (1 - v / mx);
  let o = `<defs><linearGradient id="gr" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#93c5fd"/><stop offset="1" stop-color="#6366f1"/></linearGradient>
  <linearGradient id="gh" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#f9a825"/><stop offset="1" stop-color="#ff6b6b"/></linearGradient></defs>`;
  for (let v = 0; v <= mx; v += mx / 4) o += `<line x1="${L}" x2="${W - Rr}" y1="${y(v)}" y2="${y(v)}" stroke="#eef0f5"/><text x="${L - 6}" y="${y(v) + 4}" font-size="11" fill="#9aa1b2" text-anchor="end">${f(v)}</text>`;
  D.perfil.forEach((v, i) => { const x = L + i * bw + bw * .14, sel = i === D.hora;
    o += `<rect class="bar" style="animation-delay:${i * 0.03}s" x="${x}" y="${y(v)}" width="${bw * .72}" height="${y(0) - y(v)}" rx="4" fill="url(#${sel ? 'gh' : 'gr'})"><title>${i} h: ${f(v)} Wh</title></rect>`;
    if (sel) o += `<text x="${x + bw * .36}" y="${y(v) - 6}" font-size="12" font-weight="800" fill="#ff6b6b" text-anchor="middle">${f(v)}</text>`;
    if (W > 480 || i % 2 === 0) o += `<text x="${x + bw * .36}" y="${H - 8}" font-size="11" fill="#9aa1b2" text-anchor="middle">${i}</text>`; });
  o += `<text x="4" y="11" font-size="11" fill="#9aa1b2">Wh</text>`;
  ch.innerHTML = o; }
draw(); new ResizeObserver(draw).observe(ch);
</script>
""".replace("__DATA__", json.dumps(datos)), height=760, scrolling=False)

with st.expander("📈 Desempeño del modelo (25 % de datos de prueba)"):
    st.write(f"RMSE = {metrics['RMSE']:.2f} Wh · MAE = {metrics['MAE']:.2f} Wh · "
             f"R² = {metrics['R2']:.3f} · {metrics['n_test']} registros de prueba")
st.markdown("<div class='wc-foot'>⚡ WattCast · Inteligencia Artificial Aplicada · Universidad de Lima · 2026</div>",
            unsafe_allow_html=True)
