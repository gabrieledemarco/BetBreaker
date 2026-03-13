"""
components/charts.py — Grafici Plotly per Streamlit
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import re
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


def _normalize_color(color: str) -> str:
    """
    Assicura che i colori hex abbiano il prefisso '#'.
    Se il colore è già un CSS named color o già valido, lo lascia intatto.
    """
    if not color:
        return GR
    # Se è già un hex con '#', ok
    if color.startswith('#'):
        return color
    # Se è una stringa di 6 o 8 caratteri esadecimali (RGB o RGBA), aggiungi '#'
    if re.match(r'^[0-9A-Fa-f]{6}$', color) or re.match(r'^[0-9A-Fa-f]{8}$', color):
        return '#' + color
    # Se è una stringa di 3 o 4 caratteri esadecimali (short form), aggiungi '#'
    if re.match(r'^[0-9A-Fa-f]{3}$', color) or re.match(r'^[0-9A-Fa-f]{4}$', color):
        return '#' + color
    # Altrimenti assumi sia un named color CSS valido
    return color


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


def boxplot_lap_times(driver_stats: Dict[int, Dict], driver_info: List[dict], sort_by_min_time: bool = True) -> go.Figure:
    """
    Boxplot dei lap times per pilota (solo giri validi).
    driver_stats: output di compute_session_stats['driver_stats']
    sort_by_min_time: se True, ordina i box per minimo tempo crescente
    """
    # Mappa driver_number -> nome visualizzato
    driver_names = {}
    driver_colors = {}
    for drv in driver_info:
        num = drv["driver_number"]
        driver_names[num] = drv.get("full_name", drv.get("name_acronym", str(num)))
        driver_colors[num] = _normalize_color(drv.get("team_colour", GR))
    
    # Prepara dati per boxplot
    data = []
    
    # Filtra e ordina i driver
    items = list(driver_stats.items())
    
    if sort_by_min_time:
        # Ordina per minimo tempo crescente (il pilota più veloce prima)
        items.sort(key=lambda x: min([lap["lap_duration"] for lap in x[1].get("laps", []) if lap.get("lap_duration") is not None], default=float('inf')))
    else:
        # Ordina per nome pilota
        items.sort(key=lambda x: driver_names.get(x[0], str(x[0])))
    
    for drv_num, stats in items:
        laps = stats.get("laps", [])
        durations = [lap["lap_duration"] for lap in laps if lap.get("lap_duration") is not None]
        if not durations:
            continue
        name = driver_names.get(drv_num, str(drv_num))
        color = _normalize_color(driver_colors.get(drv_num, GR))
        data.append(go.Box(
            y=durations,
            name=name,
            marker_color=color,
            boxmean='sd',
            hovertemplate=f"<b>{name}</b><br>Min: %{{y:.3f}}<br>Q1: %{{lower:.3f}}<br>Mediana: %{{median:.3f}}<br>Q3: %{{upper:.3f}}<br>Max: %{{y:.3f}}<extra></extra>"
        ))
    
    fig = go.Figure(data)
    fig.update_layout(**LAYOUT,
        title="◈ BOXPLOT LAP TIMES — Distribuzione tempi per pilota" + (" (ordinato per minimo tempo crescente)" if sort_by_min_time else ""),
        yaxis_title="Tempo (s)",
        xaxis_title="Pilota",
        height=480,
        showlegend=False,
        xaxis=dict(gridcolor="rgba(85,85,117,0.27)"),
        yaxis=dict(gridcolor="rgba(85,85,117,0.27)"),
    )
    return fig


def lap_time_scatter(lap_records: List[dict], driver_info: List[dict]) -> go.Figure:
    """
    Scatter plot lap number vs lap duration, colorato per pilota.
    lap_records: lista di giri validi (con lap_number, lap_duration, driver_number)
    """
    # Mappa driver -> nome e colore
    driver_names = {}
    driver_colors = {}
    for drv in driver_info:
        num = drv["driver_number"]
        driver_names[num] = drv.get("full_name", drv.get("name_acronym", str(num)))
        driver_colors[num] = _normalize_color(drv.get("team_colour", GR))
    
    # Raggruppa per pilota per tracce separate
    traces = []
    for drv_num in set(lap["driver_number"] for lap in lap_records):
        drv_laps = [lap for lap in lap_records if lap["driver_number"] == drv_num]
        lap_nums = [lap["lap_number"] for lap in drv_laps]
        lap_durs = [lap["lap_duration"] for lap in drv_laps]
        name = driver_names.get(drv_num, str(drv_num))
        color = _normalize_color(driver_colors.get(drv_num, GR))
        
        traces.append(go.Scatter(
            x=lap_nums,
            y=lap_durs,
            mode='markers+lines',
            name=name,
            marker_color=color,
            line_color=color,
            opacity=0.8,
            hovertemplate=f"<b>{name}</b><br>Lap %{{x}}<br>%{{y:.3f}} s<extra></extra>"
        ))
    
    fig = go.Figure(traces)
    fig.update_layout(**LAYOUT,
        title="◈ LAP TIMES PROGRESSION — Evoluzione tempi per pilota",
        xaxis_title="Numero giro",
        yaxis_title="Tempo (s)",
        height=500,
        xaxis=dict(gridcolor="rgba(85,85,117,0.27)"),
        yaxis=dict(gridcolor="rgba(85,85,117,0.27)"),
    )
    return fig


def sector_heatmap(sector_stats: Dict[int, Dict], driver_info: List[dict]) -> go.Figure:
    """
    Heatmap dei tempi medi per settore (S1, S2, S3) per pilota.
    sector_stats: output di aggregate_sector_times_by_driver
    """
    # Mappa driver_number -> nome visualizzato
    driver_names = {}
    for drv in driver_info:
        num = drv["driver_number"]
        driver_names[num] = drv.get("full_name", drv.get("name_acronym", str(num)))
    
    # Costruisci matrice: piloti (righe) x settori (colonne)
    drivers = []
    sectors = ["Sector 1", "Sector 2", "Sector 3"]
    data_matrix = []
    
    for drv_num, stats in sector_stats.items():
        driver_name = driver_names.get(drv_num, str(drv_num))
        drivers.append(driver_name)
        row = []
        sector_data = stats.get("sector_stats", {})
        for s_key in ["sector1", "sector2", "sector3"]:
            s_stats = sector_data.get(s_key)
            if s_stats and s_stats.get("mean") is not None:
                row.append(s_stats["mean"])
            else:
                row.append(None)
        data_matrix.append(row)
    
    # Crea heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data_matrix,
        x=sectors,
        y=drivers,
        colorscale='Viridis',
        hoverongaps=False,
        hoverinfo='z',
        colorbar=dict(title=dict(text="Tempo medio (s)", side="right")),
        text=[[f"{val:.3f}s" if val is not None else "N/A" for val in row] for row in data_matrix],
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    
    fig.update_layout(**LAYOUT,
        title="◈ SECTOR PERFORMANCE HEATMAP — Tempi medi per settore",
        xaxis_title="Settore",
        yaxis_title="Pilota",
        height=max(400, len(drivers) * 25 + 100),
        xaxis=dict(gridcolor="rgba(85,85,117,0.27)"),
        yaxis=dict(gridcolor="rgba(85,85,117,0.27)"),
    )
    
    return fig


def sector_contribution_bar(sector_stats: Dict[int, Dict], driver_info: List[dict]) -> go.Figure:
    """
    Grafico a barre del contributo percentuale dei settori al tempo totale per pilota.
    """
    # Mappa driver_number -> nome visualizzato
    driver_names = {}
    for drv in driver_info:
        num = drv["driver_number"]
        driver_names[num] = drv.get("full_name", drv.get("name_acronym", str(num)))
    
    # Raccogli dati per grafico a barre raggruppate
    drivers = []
    s1_contrib = []
    s2_contrib = []
    s3_contrib = []
    
    for drv_num, stats in sector_stats.items():
        contrib_stats = stats.get("contribution_stats")
        if not contrib_stats:
            continue
        drivers.append(driver_names.get(drv_num, str(drv_num)))
        s1_contrib.append(contrib_stats.get("sector1_pct_mean"))
        s2_contrib.append(contrib_stats.get("sector2_pct_mean"))
        s3_contrib.append(contrib_stats.get("sector3_pct_mean"))
    
    if not drivers:
        # Fallback: grafico vuoto
        fig = go.Figure()
        fig.update_layout(**LAYOUT,
            title="◈ SECTOR CONTRIBUTION — Nessun dato disponibile",
            height=400,
        )
        return fig
    
    # Crea tracce per ogni settore
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=drivers,
        y=s1_contrib,
        name='Sector 1',
        marker_color='#FF6B6B',
        hovertemplate="<b>%{x}</b><br>Sector 1: %{y:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=drivers,
        y=s2_contrib,
        name='Sector 2',
        marker_color='#4ECDC4',
        hovertemplate="<b>%{x}</b><br>Sector 2: %{y:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=drivers,
        y=s3_contrib,
        name='Sector 3',
        marker_color='#45B7D1',
        hovertemplate="<b>%{x}</b><br>Sector 3: %{y:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(**LAYOUT,
        title="◈ SECTOR CONTRIBUTION — Percentuale del tempo totale per settore",
        xaxis_title="Pilota",
        yaxis_title="Percentuale del giro (%)",
        barmode='group',
        height=max(450, len(drivers) * 30 + 150),
        xaxis=dict(gridcolor="rgba(85,85,117,0.27)"),
        yaxis=dict(gridcolor="rgba(85,85,117,0.27)"),
    )
    
    return fig


def sector_consistency_scatter(sector_stats: Dict[int, Dict], driver_info: List[dict]) -> go.Figure:
    """
    Scatter plot di consistenza: deviazione standard vs tempo medio per settore.
    """
    # Mappa driver_number -> nome e colore
    driver_names = {}
    driver_colors = {}
    for drv in driver_info:
        num = drv["driver_number"]
        driver_names[num] = drv.get("full_name", drv.get("name_acronym", str(num)))
        driver_colors[num] = _normalize_color(drv.get("team_colour", GR))
    
    # Crea subplot per ogni settore
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Sector 1", "Sector 2", "Sector 3"),
        horizontal_spacing=0.1,
    )
    
    for col, s_key in enumerate(["sector1", "sector2", "sector3"], 1):
        x_vals = []
        y_vals = []
        texts = []
        colors = []
        
        for drv_num, stats in sector_stats.items():
            sector_data = stats.get("sector_stats", {}).get(s_key)
            if not sector_data or sector_data.get("mean") is None or sector_data.get("std") is None:
                continue
            x_vals.append(sector_data["mean"])
            y_vals.append(sector_data["std"])
            texts.append(driver_names.get(drv_num, str(drv_num)))
            colors.append(driver_colors.get(drv_num, GR))
        
        if x_vals:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='markers+text',
                    text=texts,
                    textposition='top center',
                    marker=dict(size=10, color=colors, line=dict(width=1, color='white')),
                    hovertemplate="<b>%{text}</b><br>Tempo medio: %{x:.3f}s<br>Deviazione: %{y:.3f}s<extra></extra>",
                    showlegend=False,
                ),
                row=1, col=col
            )
    
    fig.update_layout(**LAYOUT,
        title="◈ SECTOR CONSISTENCY — Deviazione standard vs tempo medio",
        height=500,
    )
    fig.update_xaxes(title_text="Tempo medio (s)", gridcolor="rgba(85,85,117,0.27)")
    fig.update_yaxes(title_text="Deviazione standard (s)", gridcolor="rgba(85,85,117,0.27)")
    
    return fig


def sector_profile_clustering(sector_stats: Dict[int, Dict], driver_info: List[dict]) -> go.Figure:
    """
    Clustering dei piloti basato sui profili dei settori (tempi medi).
    Utilizza PCA per riduzione dimensionale e visualizzazione 2D.
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        has_sklearn = True
    except ImportError:
        has_sklearn = False
    
    # Mappa driver_number -> nome e colore
    driver_names = {}
    driver_colors = {}
    for drv in driver_info:
        num = drv["driver_number"]
        driver_names[num] = drv.get("full_name", drv.get("name_acronym", str(num)))
        driver_colors[num] = _normalize_color(drv.get("team_colour", GR))
    
    # Costruisci matrice delle features: per ogni pilota, tempi medi dei 3 settori
    features = []
    drivers = []
    colors = []
    
    for drv_num, stats in sector_stats.items():
        sector_data = stats.get("sector_stats", {})
        # Estrai tempi medi per ogni settore
        s1 = sector_data.get("sector1", {}).get("mean")
        s2 = sector_data.get("sector2", {}).get("mean")
        s3 = sector_data.get("sector3", {}).get("mean")
        
        if s1 is not None and s2 is not None and s3 is not None:
            features.append([s1, s2, s3])
            drivers.append(driver_names.get(drv_num, str(drv_num)))
            colors.append(driver_colors.get(drv_num, GR))
    
    if len(features) < 3:
        # Troppi pochi dati per clustering
        fig = go.Figure()
        fig.update_layout(**LAYOUT,
            title="◈ SECTOR PROFILE CLUSTERING — Dati insufficienti",
            height=500,
        )
        return fig
    
    features = np.array(features)
    
    if has_sklearn:
        # Standardizza le features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Applica PCA per ridurre a 2 dimensioni
        pca = PCA(n_components=2)
        components = pca.fit_transform(features_scaled)
        
        # Spiegazione varianza
        var_exp = pca.explained_variance_ratio_
        x_label = f"PC1 ({var_exp[0]*100:.1f}% var.)"
        y_label = f"PC2 ({var_exp[1]*100:.1f}% var.)"
        
        title_suffix = " (PCA)"
    else:
        # Fallback: usa le prime due dimensioni (S1 vs S2)
        components = features[:, :2]
        x_label = "Tempo medio Sector 1 (s)"
        y_label = "Tempo medio Sector 2 (s)"
        title_suffix = " (S1 vs S2)"
    
    # Crea scatter plot
    fig = go.Figure()
    
    for i, driver in enumerate(drivers):
        fig.add_trace(go.Scatter(
            x=[components[i, 0]],
            y=[components[i, 1]],
            mode='markers+text',
            text=[driver],
            textposition='top center',
            marker=dict(size=15, color=colors[i], line=dict(width=1, color='white')),
            name=driver,
            hovertemplate=f"<b>{driver}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>",
            showlegend=False,
        ))
    
    fig.update_layout(**LAYOUT,
        title=f"◈ SECTOR PROFILE CLUSTERING — Analisi profili prestazionali{title_suffix}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=600,
        xaxis=dict(gridcolor="rgba(85,85,117,0.27)"),
        yaxis=dict(gridcolor="rgba(85,85,117,0.27)"),
    )
    
    # Aggiungi linee di riferimento per la media
    fig.add_hline(y=np.mean(components[:, 1]), line_dash="dot", line_color=GR, opacity=0.5)
    fig.add_vline(x=np.mean(components[:, 0]), line_dash="dot", line_color=GR, opacity=0.5)
    
    return fig


def sector_performance_radar(sector_stats: Dict[int, Dict], driver_info: List[dict]) -> go.Figure:
    """
    Radar chart dei profili prestazionali per pilota.
    Mostra il tempo relativo rispetto al miglior tempo per ogni settore.
    """
    # Mappa driver_number -> nome e colore
    driver_names = {}
    driver_colors = {}
    for drv in driver_info:
        num = drv["driver_number"]
        driver_names[num] = drv.get("full_name", drv.get("name_acronym", str(num)))
        driver_colors[num] = _normalize_color(drv.get("team_colour", GR))
    
    # Raccoglie tempi medi per ogni settore
    sector_data = {}
    for drv_num, stats in sector_stats.items():
        sector_stats_data = stats.get("sector_stats", {})
        driver_name = driver_names.get(drv_num, str(drv_num))
        
        times = []
        for s_key in ["sector1", "sector2", "sector3"]:
            s_stats = sector_stats_data.get(s_key)
            if s_stats and s_stats.get("mean") is not None:
                times.append(s_stats["mean"])
            else:
                times.append(None)
        
        if all(t is not None for t in times):
            sector_data[driver_name] = {
                "times": times,
                "color": driver_colors.get(drv_num, GR)
            }
    
    if not sector_data:
        fig = go.Figure()
        fig.update_layout(**LAYOUT,
            title="◈ PERFORMANCE RADAR — Nessun dato settoriale completo disponibile",
            height=500,
        )
        return fig
    
    # Calcola il miglior tempo per ogni settore (minimo)
    min_times = []
    for i in range(3):  # S1, S2, S3
        sector_times = [data["times"][i] for data in sector_data.values()]
        min_times.append(min(sector_times))
    
    # Crea figura radar
    fig = go.Figure()
    
    categories = ["Sector 1", "Sector 2", "Sector 3"]
    
    for driver_name, data in sector_data.items():
        times = data["times"]
        # Calcola delta rispetto al miglior tempo (percentuale più lento)
        # Usiamo delta in secondi per chiarezza
        delta_times = [t - min_times[i] for i, t in enumerate(times)]
        
        # Per radar chart, vogliamo valori più bassi = meglio, quindi invertiamo
        # Usiamo un punteggio normalizzato: 100 - (delta * scaling factor)
        # Scaling factor: 10 secondi di delta = 100 punti
        max_delta = max(delta_times) if delta_times else 10
        scaling = 100 / max(10, max_delta * 2)  # Scaling ragionevole
        
        scores = [100 - (delta * scaling) for delta in delta_times]
        
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],  # Chiudi il poligono
            theta=categories + [categories[0]],
            name=driver_name,
            marker_color=data["color"],
            customdata=[delta_times + delta_times[:1]],  # Ripeti primo valore per chiudere
            hovertemplate=f"<b>{driver_name}</b><br>Sector: %{{theta}}<br>Score: %{{r:.1f}} pts<br>Delta S1: {delta_times[0]:.3f}s<br>Delta S2: {delta_times[1]:.3f}s<br>Delta S3: {delta_times[2]:.3f}s<extra></extra>",
            fill='toself',
            opacity=0.7,
        ))
    
    # Crea copia di LAYOUT senza legend per evitare conflitto
    layout_copy = dict(LAYOUT)
    if 'legend' in layout_copy:
        del layout_copy['legend']
    
    fig.update_layout(**layout_copy,
        title="◈ PERFORMANCE RADAR — Profili prestazionali relativi (vs miglior tempo)",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(85,85,117,0.27)",
                tickfont_color=W,
            ),
            angularaxis=dict(
                gridcolor="rgba(85,85,117,0.27)",
                linecolor=GR,
                rotation=90,  # Inizia dal top
                direction="clockwise",
            ),
            bgcolor=CARD,
        ),
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(15,15,28,0.8)",
            bordercolor=GR,
        ),
    )
    
    return fig


def sector_correlation_matrix(sector_stats: Dict[int, Dict], driver_info: List[dict]) -> go.Figure:
    """
    Matrice di correlazione tra settori e tempo totale.
    """
    # Mappa driver_number -> nome
    driver_names = {}
    for drv in driver_info:
        num = drv["driver_number"]
        driver_names[num] = drv.get("full_name", drv.get("name_acronym", str(num)))
    
    # Raccoglie dati per ogni pilota: S1, S2, S3, Totale (stima)
    data = []
    drivers = []
    
    for drv_num, stats in sector_stats.items():
        sector_stats_data = stats.get("sector_stats", {})
        
        s1 = sector_stats_data.get("sector1", {}).get("mean")
        s2 = sector_stats_data.get("sector2", {}).get("mean")
        s3 = sector_stats_data.get("sector3", {}).get("mean")
        
        if s1 is not None and s2 is not None and s3 is not None:
            total = s1 + s2 + s3  # Stima del tempo totale
            data.append([s1, s2, s3, total])
            drivers.append(driver_names.get(drv_num, str(drv_num)))
    
    if len(data) < 3:
        fig = go.Figure()
        fig.update_layout(**LAYOUT,
            title="◈ CORRELATION MATRIX — Dati insufficienti",
            height=500,
        )
        return fig
    
    # Calcola matrice di correlazione
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(data, columns=["S1", "S2", "S3", "Totale"])
    corr_matrix = df.corr()
    
    # Crea heatmap della correlazione
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        hoverongaps=False,
        text=np.round(corr_matrix.values, 3),
        texttemplate="%{text}",
        textfont={"size": 12, "color": "black"},
        colorbar=dict(title=dict(text="Correlazione", side="right")),
    ))
    
    fig.update_layout(**LAYOUT,
        title="◈ CORRELATION MATRIX — Relazioni tra settori e tempo totale",
        xaxis_title="Variabile",
        yaxis_title="Variabile",
        height=500,
        xaxis=dict(gridcolor="rgba(85,85,117,0.27)"),
        yaxis=dict(gridcolor="rgba(85,85,117,0.27)"),
    )
    
    return fig


def sector_evolution_analysis(valid_laps: List[dict], sector_stats: Dict[int, Dict], driver_info: List[dict]) -> go.Figure:
    """
    Analisi evoluzione delle prestazioni durante la sessione.
    Mostra tempi dei settori vs numero giro.
    """
    # Mappa driver_number -> nome e colore
    driver_names = {}
    driver_colors = {}
    for drv in driver_info:
        num = drv["driver_number"]
        driver_names[num] = drv.get("full_name", drv.get("name_acronym", str(num)))
        driver_colors[num] = _normalize_color(drv.get("team_colour", GR))
    
    # Filtra solo piloti presenti nei sector_stats
    included_drivers = set(sector_stats.keys())
    
    # Raccoglie dati per ogni pilota: {driver_num: {lap_number: [s1, s2, s3]}}
    driver_lap_data = {}
    
    for lap in valid_laps:
        drv_num = lap["driver_number"]
        if drv_num not in included_drivers:
            continue
        
        lap_num = lap.get("lap_number")
        s1 = lap.get("duration_sector_1")
        s2 = lap.get("duration_sector_2")
        s3 = lap.get("duration_sector_3")
        
        if lap_num is not None and s1 is not None and s2 is not None and s3 is not None:
            if drv_num not in driver_lap_data:
                driver_lap_data[drv_num] = {}
            
            driver_lap_data[drv_num][lap_num] = [s1, s2, s3]
    
    if not driver_lap_data:
        fig = go.Figure()
        fig.update_layout(**LAYOUT,
            title="◈ SESSION EVOLUTION — Nessun dato lap‑by‑lap disponibile",
            height=500,
        )
        return fig
    
    # Crea subplot: 1 riga per settore
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Sector 1", "Sector 2", "Sector 3"),
        horizontal_spacing=0.1,
        shared_yaxes=True,
    )
    
    # Limita a massimo 5 piloti per leggibilità
    max_drivers = 5
    drivers_to_plot = list(driver_lap_data.keys())[:max_drivers]
    
    for col, s_idx in enumerate([0, 1, 2], 1):
        for drv_num in drivers_to_plot:
            lap_data = driver_lap_data[drv_num]
            if not lap_data:
                continue
            
            # Ordina per numero giro
            sorted_laps = sorted(lap_data.items(), key=lambda x: x[0])
            lap_numbers = [lap[0] for lap in sorted_laps]
            sector_times = [lap[1][s_idx] for lap in sorted_laps]
            
            driver_name = driver_names.get(drv_num, str(drv_num))
            color = driver_colors.get(drv_num, GR)
            
            fig.add_trace(
                go.Scatter(
                    x=lap_numbers,
                    y=sector_times,
                    mode='lines+markers',
                    name=driver_name if col == 1 else None,  # Mostra legenda solo per primo settore
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    hovertemplate=f"<b>{driver_name}</b><br>Giro %{{x}}<br>Tempo: %{{y:.3f}}s<extra></extra>",
                    showlegend=(col == 1),
                ),
                row=1, col=col
            )
    
    # Crea copia di LAYOUT senza legend per evitare conflitto
    layout_copy = dict(LAYOUT)
    if 'legend' in layout_copy:
        del layout_copy['legend']
    
    fig.update_layout(**layout_copy,
        title="◈ SESSION EVOLUTION — Evoluzione tempi settoriali durante la sessione",
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(15,15,28,0.8)",
            bordercolor=GR,
        ),
    )
    
    fig.update_xaxes(title_text="Numero giro", gridcolor="rgba(85,85,117,0.27)")
    fig.update_yaxes(title_text="Tempo settore (s)", gridcolor="rgba(85,85,117,0.27)")
    
    return fig