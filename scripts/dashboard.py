#!/usr/bin/env python3
"""
Dashboard de visualización en tiempo real para detección MitM
CNN + Bi-LSTM sobre tráfico cifrado Wi-Fi
"""

import json
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ALERTS_FILE = "results/alerts.jsonl"

st.set_page_config(
    page_title="MitM Detector Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Detección en Tiempo Real de Ataques Man-in-the-Middle")
st.caption("Sistema basado en CNN + Bi-LSTM sobre tráfico cifrado Wi-Fi (ESPE SD)")

def read_events(path: str, max_lines: int = 5000) -> pd.DataFrame:
    """Leer eventos del archivo JSONL"""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["ts","prob_attack","threshold","state","packet_count","total_bytes","ratio_arp"])
    
    lines = p.read_text(encoding="utf-8").strip().splitlines()[-max_lines:]
    rows = []
    for ln in lines:
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue
    
    if not rows:
        return pd.DataFrame(columns=["ts","prob_attack","threshold","state","packet_count","total_bytes","ratio_arp"])
    
    df = pd.DataFrame(rows)
    df = df.sort_values("ts")
    df["datetime"] = pd.to_datetime(df["ts"], unit="s")
    return df

# Controles
col_ctrl1, col_ctrl2 = st.columns([3, 1])
with col_ctrl1:
    auto_refresh = st.toggle("🔄 Auto-actualizar", value=True, key="toggle_refresh")
with col_ctrl2:
    refresh_sec = st.slider("Intervalo (seg)", 1, 10, 1, key="slider_interval")

# Placeholders
placeholder_status = st.empty()
placeholder_metrics = st.empty()
placeholder_chart = st.empty()
placeholder_table = st.empty()

# Loop principal
iteration = 0
while True:
    iteration += 1
    df = read_events(ALERTS_FILE)
    
    if df.empty:
        placeholder_status.warning("⏳ Esperando eventos... (ejecuta `real_time_detector.py` primero)")
        time.sleep(refresh_sec)
        if not auto_refresh:
            break
        continue
    
    # Último evento
    last = df.iloc[-1]
    state = str(last.get("state", "NORMAL"))
    prob = float(last.get("prob_attack", 0.0))
    thr = float(last.get("threshold", 0.25))
    pkt_count = int(last.get("packet_count", 0))
    total_bytes = int(last.get("total_bytes", 0))
    
    # Panel de estado
    with placeholder_status.container():
        if state.upper() == "ATAQUE":
            st.error("🚨 **ESTADO: ATAQUE DETECTADO**", icon="🚨")
        else:
            st.success("✅ **ESTADO: NORMAL**", icon="✅")
    
    # Métricas principales
    with placeholder_metrics.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Probabilidad de Ataque", f"{prob:.3f}", help="Salida del modelo (0 a 1)")
        col2.metric("Umbral", f"{thr:.2f}")
        col3.metric("Paquetes (ventana)", f"{pkt_count}")
        col4.metric("Bytes totales", f"{total_bytes:,}")
    
    # Gráfico histórico
    tail = df.tail(300).copy()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Probabilidad de Ataque vs Umbral", "Paquetes por Ventana"),
        row_heights=[0.6, 0.4]
    )
    
    # Gráfico 1: Probabilidad
    fig.add_trace(
        go.Scatter(
            x=tail["datetime"],
            y=tail["prob_attack"],
            mode="lines+markers",
            name="Prob. Ataque",
            line=dict(color="red", width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=tail["datetime"],
            y=tail["threshold"],
            mode="lines",
            name="Umbral",
            line=dict(color="orange", dash="dash", width=2)
        ),
        row=1, col=1
    )
    
    # Gráfico 2: Paquetes
    fig.add_trace(
        go.Bar(
            x=tail["datetime"],
            y=tail["packet_count"],
            name="Paquetes",
            marker=dict(color="steelblue")
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Tiempo", row=2, col=1)
    fig.update_yaxes(title_text="Probabilidad", row=1, col=1)
    fig.update_yaxes(title_text="Paquetes", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode="x unified"
    )
    
    # Usar key único para evitar duplicados
    placeholder_chart.plotly_chart(fig, key=f"chart_{iteration}")
    
    # Tabla de últimos eventos
    with placeholder_table.container():
        st.subheader("📋 Últimos 10 Eventos")
        display_df = tail[["datetime", "state", "prob_attack", "packet_count", "total_bytes", "ratio_arp"]].tail(10).copy()
        display_df = display_df.rename(columns={
            "datetime": "Timestamp",
            "state": "Estado",
            "prob_attack": "Prob. Ataque",
            "packet_count": "Paquetes",
            "total_bytes": "Bytes",
            "ratio_arp": "Ratio ARP"
        })
        st.dataframe(display_df, hide_index=True, key=f"table_{iteration}")
    
    if not auto_refresh:
        break
    
    time.sleep(refresh_sec)