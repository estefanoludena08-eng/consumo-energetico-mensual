"""WattCast - Prototipo de pronóstico del consumo mensual de electrodomésticos.

Datos: Candanedo, Feldheim y Deramaix (2017), "Appliances Energy Prediction".
Modelo: Random Forest (100 árboles), el mejor modelo según la comparación hecha en Orange.
El modelo predice el consumo de cada intervalo de 10 minutos; la app suma las predicciones de las
24 horas del día y las multiplica por 30 días para estimar el consumo mensual.
Código generado con apoyo de inteligencia artificial (Claude, Anthropic) y revisado por el equipo.

Ejecutar:  streamlit run app.py   (el archivo energydata_complete.csv debe estar en la misma carpeta)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA = Path(__file__).parent / "energydata_complete.csv"
ROOMS = [1, 2, 3, 4, 5, 7, 8, 9]          # ambientes interiores (T6/RH_6 están fuera de la casa)
LOW, HIGH = 400, 625                      # kWh/mes: mediana y ~percentil 90 del consumo diario x 30 días


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
        return "BAJO", "#2e7d32"
    if kwh_mes <= HIGH:
        return "MEDIO", "#f9a825"
    return "ALTO", "#c62828"


st.set_page_config(page_title="WattCast", page_icon="⚡", layout="centered")

# ---------- Estilos: título centrado y animación de entrada (solo CSS) ----------
ESTILOS = """
<style>
.wc-head{text-align:center;margin:0 0 8px}
.wc-title{display:inline-block;font-size:3.4rem;font-weight:800;line-height:1.1;margin:0;letter-spacing:-.02em;
  background:linear-gradient(90deg,#f9a825,#ff4b4b 55%,#c62828);-webkit-background-clip:text;background-clip:text;color:transparent!important;-webkit-text-fill-color:transparent;padding:0!important}
.wc-bolt-ico{font-size:3rem;vertical-align:.15em;margin-right:.15em}
.wc-head a{display:none!important}
.wc-sub{color:#808495;font-size:1rem;margin:6px auto 0;max-width:560px}
.wc-intro{position:fixed;inset:0;z-index:999999;display:flex;flex-direction:column;align-items:center;justify-content:center;
  background:radial-gradient(circle at 50% 45%,#1d2433 0%,#0b0e14 70%);
  animation:wcOut .7s ease-in 3.1s forwards}
.wc-bolt{width:120px;height:120px;filter:drop-shadow(0 0 0 #f9c80e);animation:wcGlow 1.2s ease-in-out .9s 2 alternate}
.wc-bolt path{fill:transparent;stroke:#f9c80e;stroke-width:3;stroke-linejoin:round;stroke-dasharray:260;stroke-dashoffset:260;
  animation:wcDraw .9s ease-out forwards,wcFill .5s ease-out .8s forwards}
.wc-ring{position:absolute;width:120px;height:120px;border-radius:50%;border:2px solid #f9c80e;opacity:0;
  animation:wcRing 1.4s ease-out 1s 2}
.wc-ring.r2{animation-delay:1.4s}
.wc-name{display:flex;gap:2px;margin-top:22px;font-size:3.6rem;font-weight:800;letter-spacing:-.02em;color:#fff}
.wc-name b{opacity:0;transform:translateY(24px);animation:wcUp .5s cubic-bezier(.2,.8,.2,1.2) forwards}
.wc-name b.y{color:#f9c80e}
.wc-tag{margin-top:6px;color:#a3a8b8;font-size:1.05rem;opacity:0;animation:wcFade .6s ease 1.9s forwards}
.wc-bar{width:220px;height:6px;border-radius:3px;background:#262b36;margin-top:22px;overflow:hidden}
.wc-bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,#2e7d32,#f9a825,#ff4b4b);animation:wcCharge 2.2s ease-in-out .5s forwards}
@keyframes wcDraw{to{stroke-dashoffset:0}}
@keyframes wcFill{to{fill:#f9c80e}}
@keyframes wcGlow{to{filter:drop-shadow(0 0 22px #f9c80e)}}
@keyframes wcRing{0%{opacity:.8;transform:scale(1)}100%{opacity:0;transform:scale(2.6)}}
@keyframes wcUp{to{opacity:1;transform:none}}
@keyframes wcFade{to{opacity:1}}
@keyframes wcCharge{to{width:100%}}
@keyframes wcOut{to{opacity:0;transform:translateY(-30px);visibility:hidden}}
@media (prefers-reduced-motion:reduce){.wc-intro{display:none}}
</style>
"""
st.markdown(ESTILOS, unsafe_allow_html=True)

if "intro_vista" not in st.session_state:          # la animación solo aparece al abrir la app
    st.session_state.intro_vista = True
    letras = "".join(
        f"<b class='{'y' if i >= 4 else ''}' style='animation-delay:{1.0 + i * 0.08:.2f}s'>{c}</b>"
        for i, c in enumerate("WattCast"))
    st.markdown(
        "<div class='wc-intro'>"
        "<div style='position:relative;display:flex;align-items:center;justify-content:center'>"
        "<span class='wc-ring'></span><span class='wc-ring r2'></span>"
        "<svg class='wc-bolt' viewBox='0 0 64 64'><path d='M36 4 L12 36 H29 L25 60 L52 26 H35 Z'/></svg></div>"
        f"<div class='wc-name'>{letras}</div>"
        "<div class='wc-tag'>Pronóstico inteligente del consumo de tu hogar</div>"
        "<div class='wc-bar'><i></i></div></div>",
        unsafe_allow_html=True)

st.markdown(
    "<div class='wc-head'><div><span class='wc-bolt-ico'>⚡</span><span class='wc-title'>WattCast</span></div>"
    "<p class='wc-sub'>Pronóstico del consumo mensual de los electrodomésticos de una vivienda "
    "con Random Forest (datos de Candanedo et al., 2017).</p></div>",
    unsafe_allow_html=True)

df = load_data()
with st.spinner("Entrenando el modelo (solo la primera vez)..."):
    model, cols, metrics = train(df)
med = df.median(numeric_only=True)

st.subheader("1. Ingrese las condiciones")
c1, c2 = st.columns(2)
with c1:
    hora = st.slider("Hora del día", 0, 23, 18)
    lights = st.select_slider("Consumo de las luces (Wh)", options=list(range(0, 80, 10)), value=10)
    t_in = st.slider("Temperatura interior promedio (°C)", 15.0, 26.0, 20.0, 0.5)
    rh_in = st.slider("Humedad interior promedio (%)", 28.0, 60.0, 42.0, 1.0)
with c2:
    t_out = st.slider("Temperatura exterior (°C)", -5.0, 26.0, 7.0, 0.5)
    rh_out = st.slider("Humedad exterior (%)", 24.0, 100.0, 84.0, 1.0)
    wind = st.slider("Velocidad del viento (m/s)", 0.0, 14.0, 3.5, 0.5)
with st.expander("Opciones avanzadas (por defecto, valores típicos del conjunto de datos)"):
    press = st.number_input("Presión atmosférica (mm Hg)", 729.0, 773.0, float(round(med["Press_mm_hg"], 1)))
    vis = st.number_input("Visibilidad (km)", 1.0, 66.0, float(med["Visibility"]))
    dew = st.number_input("Punto de rocío (°C)", -7.0, 16.0, float(round(med["Tdewpoint"], 1)))

if st.button("🔮 Predecir consumo", type="primary", use_container_width=True):
    rows = [build_row(df, h, lights, t_in, rh_in, t_out, rh_out, wind, press, vis, dew) for h in range(24)]
    wh10 = model.predict(pd.DataFrame(rows)[cols])          # Wh en 10 minutos, para cada hora
    wh_hora = wh10 * 6                                       # Wh en cada hora
    kwh_dia = wh_hora.sum() / 1000
    kwh_mes = kwh_dia * 30
    txt, color = nivel(kwh_mes)
    st.subheader("2. Resultado")
    st.markdown(
        f"<div style='padding:18px;border-radius:10px;background:{color}1a;border:2px solid {color}'>"
        f"<div style='font-size:15px'>Consumo mensual estimado de los electrodomésticos</div>"
        f"<div style='font-size:40px;font-weight:700'>{kwh_mes:.0f} kWh/mes</div>"
        f"<div style='font-size:18px;color:{color};font-weight:700'>Nivel de consumo: {txt}</div>"
        f"<div style='font-size:15px;margin-top:6px'>Equivale a {kwh_dia:.1f} kWh por día. A las {hora} h: "
        f"{wh10[hora]:.0f} Wh cada 10 minutos ({wh_hora[hora] / 1000:.2f} kWh en esa hora).</div></div>",
        unsafe_allow_html=True)
    st.caption(f"Bajo: hasta {LOW} kWh/mes (mediana) · Medio: {LOW}–{HIGH} kWh/mes · "
               f"Alto: más de {HIGH} kWh/mes (≈10 % de los días con más consumo). "
               "Se calcula con 30 días iguales, con las condiciones ingresadas durante las 24 horas.")

    perfil = pd.DataFrame({"Hora": range(24), "Consumo por hora (Wh)": wh_hora}).set_index("Hora")
    st.markdown("**Consumo estimado por hora con las mismas condiciones**")
    st.bar_chart(perfil)

with st.expander("Desempeño del modelo (25 % de datos de prueba)"):
    st.write(f"RMSE = {metrics['RMSE']:.2f} Wh · MAE = {metrics['MAE']:.2f} Wh · "
             f"R² = {metrics['R2']:.3f} · {metrics['n_test']} registros de prueba")
