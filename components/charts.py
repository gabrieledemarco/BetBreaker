"""
components/charts.py — Grafici Plotly per Streamlit
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict
from core.analysis import Pred, Multipla

# Palette
BG    = "#07070f"
CARD  = "#0f0f1c"
G     = "#00e676"
R     = "#ff1744"
Y     = "#ffd740"
B     = "#448aff"
P     = "#d500f9"
O     = "#ff6d00"
C     = "#18ffff"
W     = "#f0f0f0"
GR    = "#555575"

LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=CARD,
    font_color=W, font_family="Inter, DejaVu Sans",
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor=GR),
)


def color_edge(e: float) -> str:
    if e > 0.04: return G
    if e < -0.05: return R
    return Y


def edge_chart(preds: List[Pred]) -> go.Figure:
    """Grafico orizzontale edge per tutte le selezioni."""
    ps = sorted(preds, key=lambda p: p.edge)
    labels = [f"[{p.entry.mkt.value[:4].upper()}] {p.label}" for p in ps]
    edges  = [p.edge_pct for p in ps]
    colors = [color_edge(p.edge) for p in ps]
    texts  = [f"{p.edge_pct:+.1f}%  Q:{p.quota:.2f}→{p.fair_q:.2f}" for p in ps]

    fig = go.Figure(go.Bar(
        x=edges, y=labels, orientation='h',
        marker_color=colors,
        text=texts, textposition='outside',
        textfont_size=10,
        hovertemplate="<b>%{y}</b><br>Edge: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=W, line_width=1.5)
    fig.update_layout(**LAYOUT,
        title="◈ EDGE ANALYSIS — Tutte le selezioni",
        xaxis_title="Edge % (Prob. ML Ensemble vs Implicita Bookmaker)",
        height=max(350, len(ps)*28+80),
        xaxis=dict(gridcolor="rgba(85,85,117,0.27)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def prob_comparison_chart(preds: List[Pred], top_n: int = 10) -> go.Figure:
    """Gruppi di barre: Implicita vs LR vs RF vs Ensemble, per top-N."""
    ps = sorted(preds, key=lambda p: p.edge, reverse=True)[:top_n]
    labels = [p.label[:22] for p in ps]

    fig = go.Figure()
    bar_data = [
        ("Implicita Bookmaker", [p.impl*100 for p in ps], GR),
        ("Logistic Regression", [p.lr*100  for p in ps], B),
        ("Random Forest",       [p.rf*100  for p in ps], O),
        ("Ensemble",            [p.ens*100 for p in ps], G),
    ]
    for name, vals, col in bar_data:
        fig.add_trace(go.Bar(name=name, x=labels, y=vals,
                             marker_color=col, opacity=0.88))

    # Annotazioni edge
    for i,p in enumerate(ps):
        fig.add_annotation(x=i, y=p.ens*100+3,
                           text=f"{p.edge_pct:+.0f}%",
                           showarrow=False, font_color=color_edge(p.edge),
                           font_size=11, font_family="Inter")

    fig.update_layout(**LAYOUT,
        title="◈ TOP SELEZIONI — Confronto Probabilità Modelli",
        barmode='group', height=420,
        xaxis=dict(tickangle=-20, gridcolor="rgba(85,85,117,0.27)"),
        yaxis=dict(title="%", gridcolor="rgba(85,85,117,0.27)", range=[0,110]),
    )
    return fig


def session_heatmap(sessions: Dict[str, Dict[str, float]],
                    grid: Dict[str, int]) -> go.Figure:
    """Heatmap gap dal leader per sessione."""
    if not sessions:
        fig = go.Figure()
        fig.update_layout(**LAYOUT, title="Nessun dato sessione disponibile")
        return fig

    sess_order = [s for s in ['FP1','FP2','FP3','Sprint','Quali'] if s in sessions]
    drivers    = sorted(grid.keys(), key=lambda d: grid.get(d, 99))[:16]
    z     = [[min(sessions[s].get(d, 5.0), 4.5) for s in sess_order] for d in drivers]
    text  = [[f"{sessions[s].get(d, 99):.2f}s" if sessions[s].get(d,99)<9 else "—"
              for s in sess_order] for d in drivers]
    ylabels = [f"P{grid.get(d,'?')} {d}" for d in drivers]

    fig = go.Figure(go.Heatmap(
        z=z, x=sess_order, y=ylabels,
        text=text, texttemplate="%{text}",
        colorscale=[[0,G],[0.5,Y],[1,R]],
        zmin=0, zmax=4.0,
        showscale=True,
        hovertemplate="<b>%{y}</b> — %{x}<br>Gap: %{z:.3f}s<extra></extra>",
    ))
    fig.update_layout(**LAYOUT,
        title="◈ GAP AL LEADER — Heatmap Sessioni  (verde=veloce, rosso=lento)",
        height=max(380, len(drivers)*28+80),
        xaxis=dict(side='top'),
    )
    return fig


def feature_importance_chart(fi: pd.Series) -> go.Figure:
    vals   = fi.values * 100
    labels = fi.index.tolist()
    colors = [G if v > 15 else B if v > 7 else GR for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation='h',
        marker_color=colors,
        text=[f"{v:.1f}%" for v in vals],
        textposition='outside',
    ))
    fig.update_layout(**LAYOUT,
        title="◈ Feature Importance (RF — H2H Differenziale)",
        height=380,
        xaxis=dict(title="%", gridcolor="rgba(85,85,117,0.27)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def safety_car_chart(sc_hist: List[int], circuit_name: str,
                      sc_pred: "Pred|None" = None) -> go.Figure:
    if not sc_hist:
        fig = go.Figure()
        fig.update_layout(**LAYOUT, title="Storico SC non disponibile")
        return fig

    years  = list(range(2026-len(sc_hist), 2026))
    colors = [G if v else R for v in sc_hist]
    rate   = sum(sc_hist)/len(sc_hist)*100

    fig = go.Figure(go.Bar(
        x=years, y=sc_hist,
        marker_color=colors,
        text=["SC" if v else "No SC" for v in sc_hist],
        textposition='inside',
        hovertemplate="%{x}: %{text}<extra></extra>",
    ))
    fig.add_hline(y=rate/100, line_dash='dash', line_color=C,
                  annotation_text=f"Media {rate:.0f}%", annotation_font_color=C)
    title = f"◈ SAFETY CAR — {circuit_name}"
    if sc_pred:
        title += f"  |  Q:{sc_pred.quota:.2f}  Edge:{sc_pred.edge_pct:+.1f}%"
    fig.update_layout(**LAYOUT, title=title, height=280,
        yaxis=dict(tickvals=[0,1], ticktext=['No','Sì']),
        xaxis=dict(gridcolor="rgba(85,85,117,0.27)"),
    )
    return fig


def auc_chart(cv: Dict[str,float]) -> go.Figure:
    names = ["LR","RF","GBM"]
    aucs  = [cv.get('h2h_lr',0.70), cv.get('h2h_rf',0.70), cv.get('h2h_gb',0.70)]
    fig   = go.Figure(go.Bar(
        x=names, y=aucs,
        marker_color=[B,O,P],
        text=[f"{a:.3f}" for a in aucs],
        textposition='outside',
    ))
    fig.add_hline(y=0.70, line_dash='dash', line_color=Y, annotation_text="Base 0.70")
    fig.update_layout(**LAYOUT, title="◈ AUC-ROC (5-fold CV)",
        height=260, yaxis=dict(range=[0.60,0.82], gridcolor="rgba(85,85,117,0.27)"))
    return fig


def multiples_table(mults: List[Multipla]) -> go.Figure:
    if not mults:
        return go.Figure()

    risk_icons = ['⭐⭐','⭐⭐','⭐⭐⭐','⭐⭐⭐','⭐⭐⭐⭐⭐']
    rows = []
    for i,m in enumerate(mults):
        picks_str = "<br>".join([f"● {p.label} Q={p.quota:.2f} ({p.edge_pct:+.0f}%)" for p in m.picks])
        rows.append([m.label, picks_str, f"{m.q_comb:.2f}x",
                     f"€{m.win:.2f}", risk_icons[min(i,4)]])

    colors_row = [["#0a2a0a","#1a180a","#1a140a","#1a1008","#2a0808"]]
    win_colors  = [G if m.win>=25 else Y if m.win>=12 else O for m in mults]

    fig = go.Figure(go.Table(
        header=dict(values=["Multipla","Selezioni","Q.Comb.","Vincita €2","Rischio"],
                    fill_color=CARD, font_color=W, font_size=12, align='left',
                    line_color=GR),
        cells=dict(
            values=list(zip(*rows)) if rows else [[],[],[],[]],
            fill_color=[CARD,CARD,CARD,
                        [[G if m.win>=25 else Y if m.win>=12 else O for m in mults]],
                        CARD],
            font_color=[W,W,W,["black"]*len(mults),W],
            font_size=[11,10,12,13,11],
            align=['left','left','center','center','center'],
            line_color=GR,
            height=42,
        )
    ))
    fig.update_layout(**LAYOUT, title="◈ 5 MULTIPLE ML-VALIDATED", height=60+len(mults)*48)
    return fig


def quali_gap_bar(quali: Dict[str, dict]) -> go.Figure:
    """Bar chart gap qualifiche dai dati reali API."""
    if not quali:
        return go.Figure()
    items = sorted(quali.items(), key=lambda x: x[1].get('position',99))[:18]
    drivers = [d for d,_ in items]
    gaps    = [min(v.get('gap_to_leader',5),5) for _,v in items]
    pos_str = [f"P{v.get('position','?')}" for _,v in items]
    colors  = [G if g<0.3 else B if g<0.8 else Y if g<1.5 else R for g in gaps]

    fig = go.Figure(go.Bar(
        x=drivers, y=gaps, marker_color=colors,
        text=[f"{g:.3f}s" for g in gaps], textposition='outside',
        customdata=pos_str,
        hovertemplate="<b>%{x}</b> (%{customdata})<br>Gap: %{y:.3f}s<extra></extra>",
    ))
    fig.update_layout(**LAYOUT,
        title="◈ GAP QUALIFICHE — Dati reali API",
        yaxis_title="Gap (s)", height=320,
        xaxis=dict(gridcolor="rgba(85,85,117,0.27)"),
        yaxis=dict(gridcolor="rgba(85,85,117,0.27)"),
    )
    return fig