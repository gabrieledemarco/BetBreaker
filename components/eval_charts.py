"""
components/eval_charts.py — Grafici Plotly per la valutazione ex-post
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional
from core.results import PredictionRecord
from ml.evaluator import EvalMetrics, calibration_data, roi_by_edge_threshold, edge_vs_accuracy_bins

BG = "#07070f"; CARD = "#0f0f1c"; LINE = "#181830"
G = "#00e676"; R = "#ff1744"; Y = "#ffd740"
B = "#448aff"; P = "#d500f9"; O = "#ff6d00"; C = "#18ffff"; W = "#f0f0f0"; GR = "#555575"

LAYOUT = dict(paper_bgcolor=BG, plot_bgcolor=CARD, font_color=W,
              font_family="Inter, sans-serif", margin=dict(l=10,r=10,t=44,b=10),
              legend=dict(bgcolor="rgba(0,0,0,.3)", bordercolor=GR))


def reliability_diagram(records: List[PredictionRecord]) -> go.Figure:
    """
    Reliability diagram (calibration curve).
    Una curva perfettamente calibrata segue la diagonale.
    Sopra = sottostima, sotto = sovrastima.
    """
    prob_pred, prob_true, counts = calibration_data(records, n_bins=10)
    if len(prob_pred) == 0:
        fig = go.Figure()
        fig.update_layout(**LAYOUT, title="Dati insufficienti per la curva di calibrazione")
        return fig

    fig = go.Figure()
    # Diagonale perfetta
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode='lines',
        line=dict(color=GR, dash='dash', width=1.5),
        name='Calibrazione perfetta', showlegend=True))
    # Curva modello ML
    fig.add_trace(go.Scatter(
        x=prob_pred, y=prob_true, mode='lines+markers',
        line=dict(color=G, width=2.5), marker=dict(size=10, color=G),
        name='Modello ML Ensemble',
        text=[f"n={c}" for c in counts], textposition='top center',
        hovertemplate="Pred: %{x:.2f}<br>Reale: %{y:.2f}<extra></extra>"))
    # Istogramma predizioni (asse secondario)
    fig.add_trace(go.Bar(
        x=prob_pred, y=counts/counts.max()*0.15,
        marker_color=B, opacity=0.4, name='Distribuzione predizioni',
        yaxis='y'))

    # Shade zone: sopra diag = sottostima (buono per bettor), sotto = sovrastima
    fig.add_shape(type='rect', x0=0, x1=1, y0=0, y1=1,
                  fillcolor='rgba(255,215,64,0.03)', line_width=0)

    fig.update_layout(**LAYOUT,
        title="◈ RELIABILITY DIAGRAM — Calibrazione del modello<br>"
              "<sup>Sopra la diagonale = modello sottostima (conservativo) · "
              "Sotto = sovrastima (ottimista)</sup>",
        xaxis=dict(title="Probabilità predetta (ML)", range=[0,1], gridcolor="rgba(85,85,117,0.27)"),
        yaxis=dict(title="Frequenza osservata (win rate reale)", range=[0,1], gridcolor="rgba(85,85,117,0.27)"),
        height=420)
    return fig


def roi_threshold_chart(records: List[PredictionRecord]) -> go.Figure:
    """ROI simulato per ogni soglia edge. Trova il punto ottimale."""
    df = roi_by_edge_threshold(records)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**LAYOUT, title="Dati insufficienti")
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # ROI
    fig.add_trace(go.Scatter(
        x=df['edge_threshold'], y=df['roi_pct'],
        mode='lines', line=dict(color=G, width=2.5),
        name='ROI %', fill='tozeroy', fillcolor='rgba(0,230,118,.08)',
        hovertemplate="Soglia: %{x:.0f}%<br>ROI: %{y:.1f}%<extra></extra>"))
    # N scommesse
    fig.add_trace(go.Bar(
        x=df['edge_threshold'], y=df['n_bets'],
        marker_color=B, opacity=0.35, name='N scommesse'),
        secondary_y=True)
    # Win rate
    fig.add_trace(go.Scatter(
        x=df['edge_threshold'], y=df['win_rate_pct'],
        mode='lines', line=dict(color=Y, width=1.5, dash='dot'),
        name='Win rate %',
        hovertemplate="Soglia: %{x:.0f}%<br>Win rate: %{y:.1f}%<extra></extra>"))

    fig.add_hline(y=0, line_color=R, line_width=1, line_dash='dash')

    # Evidenzia max ROI
    if not df.empty:
        best = df.loc[df['roi_pct'].idxmax()]
        fig.add_vline(x=best['edge_threshold'], line_color=G, line_dash='dot',
                      annotation_text=f"Soglia ottimale: {best['edge_threshold']:.0f}%  ROI={best['roi_pct']:.1f}%",
                      annotation_font_color=G)

    fig.update_layout(**LAYOUT,
        title="◈ ROI SIMULATO per soglia edge — trova la soglia ottimale di puntata",
        xaxis=dict(title="Soglia minima edge (%)", gridcolor="rgba(85,85,117,0.27)"),
        yaxis=dict(title="ROI %", gridcolor="rgba(85,85,117,0.27)"),
        yaxis2=dict(title="N scommesse", gridcolor="rgba(0,0,0,0)"),
        height=380)
    return fig


def edge_accuracy_chart(records: List[PredictionRecord]) -> go.Figure:
    """
    Per fascia di edge: confronta win-rate atteso (ML) vs osservato (reale).
    Se le barre sono allineate → il modello è ben calibrato sull'edge.
    """
    df = edge_vs_accuracy_bins(records, n_bins=8)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**LAYOUT, title="Dati insufficienti")
        return fig

    x = df['Edge range']
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=df['Prob ML avg %'],   name='Prob ML media %',   marker_color=B,  opacity=0.85))
    fig.add_trace(go.Bar(x=x, y=df['Win rate %'],       name='Win rate reale %',  marker_color=G,  opacity=0.85))
    fig.add_trace(go.Bar(x=x, y=df['Prob impl avg %'],  name='Prob implicita bk', marker_color=GR, opacity=0.60))
    # ROI per fascia
    fig.add_trace(go.Scatter(
        x=x, y=df['ROI %'], mode='lines+markers',
        line=dict(color=Y, width=2), marker=dict(size=8),
        name='ROI % (fascia)', yaxis='y2'))

    fig.add_hline(y=0, line_color=R, line_dash='dash', line_width=1)
    fig.update_layout(**LAYOUT,
        title="◈ EDGE vs WIN RATE REALE — Le barre allineate indicano buona calibrazione",
        barmode='group',
        xaxis=dict(title="Fascia Edge", tickangle=-20, gridcolor="rgba(85,85,117,0.27)"),
        yaxis=dict(title="%", gridcolor="rgba(85,85,117,0.27)"),
        yaxis2=dict(title="ROI %", overlaying='y', side='right'),
        height=420)
    return fig


def outcome_scatter(records: List[PredictionRecord]) -> go.Figure:
    """
    Scatter plot: asse X = prob ML, asse Y = quota, colore = outcome.
    Mostra dove il modello aggiunge valore rispetto al bookmaker.
    """
    ev = [r for r in records if r.outcome is not None]
    if len(ev) < 3:
        fig = go.Figure()
        fig.update_layout(**LAYOUT, title="Dati insufficienti")
        return fig

    won  = [r for r in ev if r.outcome]
    lost = [r for r in ev if not r.outcome]

    fig = go.Figure()
    for subset, color, name, sym in [
        (won, G, '✅ Vinto', 'circle'),
        (lost, R, '❌ Perso', 'x'),
    ]:
        if subset:
            fig.add_trace(go.Scatter(
                x=[r.ensemble_prob*100 for r in subset],
                y=[r.quota             for r in subset],
                mode='markers',
                marker=dict(color=color, size=10, symbol=sym,
                            line=dict(color='white', width=0.5)),
                name=name,
                text=[f"{r.label}<br>Edge: {r.edge*100:+.1f}%" for r in subset],
                hovertemplate="<b>%{text}</b><br>Prob ML: %{x:.1f}%<br>Quota: %{y:.2f}<extra></extra>",
            ))

    # Linea fair quota (y = 1/x*100)
    xx = np.linspace(5, 95, 100)
    yy = 100 / xx
    fig.add_trace(go.Scatter(x=xx, y=yy, mode='lines',
        line=dict(color=Y, dash='dot', width=1.2),
        name='Fair quota (1/prob)',
        hoverinfo='skip'))

    fig.update_layout(**LAYOUT,
        title="◈ SCATTER PREDIZIONI — Prob ML vs Quota bookmaker<br>"
              "<sup>Sopra la linea gialla = quota > fair value (potenziale valore)</sup>",
        xaxis=dict(title="Prob. ML Ensemble (%)", range=[0,100], gridcolor="rgba(85,85,117,0.27)"),
        yaxis=dict(title="Quota bookmaker", range=[1,max(r.quota for r in ev)+0.5],
                   gridcolor="rgba(85,85,117,0.27)"),
        height=420)
    return fig


def metrics_kpi_cards(m: EvalMetrics) -> str:
    """Ritorna HTML con le metriche principali come cards.
    NOTA: usa stringhe inline senza indentazione per evitare che Streamlit
    interpreti l'HTML come blocchi di codice Markdown (regola dei 4 spazi).
    """
    def card(title, value, color, sub=""):
        s = "background:#0f0f1c;border:1px solid #181830;border-radius:12px;padding:16px 20px;text-align:center;border-top:3px solid " + color
        t = "color:#888;font-size:.75rem;text-transform:uppercase;letter-spacing:.08em"
        v = f"color:{color};font-size:1.8rem;font-weight:900;margin:4px 0"
        u = "color:#666;font-size:.72rem"
        return (f'<div style="{s}">'
                f'<div style="{t}">{title}</div>'
                f'<div style="{v}">{value}</div>'
                f'<div style="{u}">{sub}</div>'
                f'</div>')

    roi_color  = G if m.roi_green > 0 else R
    acc_color  = G if m.accuracy > 0.55 else Y if m.accuracy > 0.45 else R
    bs_color   = G if m.brier_score < 0.20 else Y if m.brier_score < 0.25 else R
    bias_color = G if abs(m.overconf_bias) < 0.03 else Y if abs(m.overconf_bias) < 0.07 else R

    g4 = "display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:12px 0"
    g3 = "display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:0 0 16px 0"

    row1 = (card("Accuracy",          f"{m.accuracy*100:.1f}%",          acc_color,  f"N={m.n_evaluated}") +
            card("Brier Score",        f"{m.brier_score:.4f}",            bs_color,   "↓ meglio, baseline=0.25") +
            card("ROI 🟢",            f"{m.roi_green:+.1f}%",            roi_color,  "stake unitario edge>4%") +
            card("Bias calibrazione",  f"{m.overconf_bias*100:+.1f}%",   bias_color, "↑ottimista ↓conservativo"))

    row2 = (card("ROI tutte",  f"{m.roi_all:+.1f}%",             G if m.roi_all>0 else R,             "ogni selezione") +
            card("ROI edge+",  f"{m.roi_positive_edge:+.1f}%",   G if m.roi_positive_edge>0 else R,   "solo edge≥0") +
            card("AUC-ROC",    f"{m.roc_auc:.3f}",               G if m.roc_auc>0.60 else Y,           "ex-post"))

    return f'<div style="{g4}">{row1}</div><div style="{g3}">{row2}</div>'


def cs_update_chart(cs_estimated: dict, cs_default: dict) -> go.Figure:
    """Confronta constructor strength default vs stimata dai dati reali."""
    if not cs_estimated:
        fig = go.Figure()
        fig.update_layout(**LAYOUT, title="Dati insufficienti per stima CS")
        return fig

    drivers = sorted(cs_estimated.keys(), key=lambda d: -cs_estimated[d])
    x = drivers[:12]
    y_est  = [cs_estimated.get(d, 0) for d in x]
    y_def  = [cs_default.get(d, 0.5) for d in x]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y_def,  name='Default (config)', marker_color=GR, opacity=0.7))
    fig.add_trace(go.Bar(x=x, y=y_est,  name='Stimata da risultati reali', marker_color=G, opacity=0.9))
    fig.update_layout(**LAYOUT,
        title="◈ CONSTRUCTOR STRENGTH — Config vs Stima empirica da risultati reali",
        barmode='group',
        xaxis=dict(gridcolor="rgba(85,85,117,0.27)"),
        yaxis=dict(title="Strength (0–1)", range=[0,1.1], gridcolor="rgba(85,85,117,0.27)"),
        height=340)
    return fig