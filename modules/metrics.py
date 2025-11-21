"""
Métricas y visualizaciones
"""
import pandas as pd
import plotly.express as px
import csv
from datetime import datetime

def log_metrics(data):
    """Guarda métricas en CSV."""
    row = {
        "timestamp": datetime.now().isoformat(),
        **data
    }
    file_exists = __import__('os').path.exists("metrics_log.csv")
    with open("metrics_log.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def generate_plots(metrics):
    """Genera gráficos Plotly."""
    # Flujo PRISMA
    df_flow = pd.DataFrame({
        "Etapa": ["Inicial", "Filtrados", "Dedup", "Relevantes"],
        "Cantidad": [
            metrics["total"],
            metrics["after_filter"],
            metrics["after_dedup"],
            metrics["relevant"]
        ]
    })
    fig_prisma = px.funnel(df_flow, x="Cantidad", y="Etapa", title="Flujo PRISMA")

    # Tiempos
    df_time = pd.DataFrame({
        "Fase": ["Búsqueda", "Filtro", "Dedup", "Cribado", "Síntesis"],
        "Tiempo (s)": [
            metrics["t_search"], metrics["t_filter"], metrics["t_dedup"],
            metrics["t_screen"], metrics["t_synth"]
        ]
    })
    fig_time = px.bar(df_time, x="Fase", y="Tiempo (s)", title="Tiempos por Fase")

    return {
        "prisma": fig_prisma.to_html(include_plotlyjs="cdn"),
        "time": fig_time.to_html(include_plotlyjs="cdn")
    }