"""
app.py — F1 ML Odds Analyzer · Streamlit Web App
"""
import streamlit as st
import pandas as pd
import numpy as np
import time, json, os
from pathlib import Path

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BetBreaker F1 — ML Odds Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark background */
  .stApp { background: #07070f; color: #f0f0f0; }
  section[data-testid="stSidebar"] { background: #0a0a18 !important; }
  .block-container { padding-top: 1.2rem; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #0f0f1c; border: 1px solid #181830;
    border-radius: 10px; padding: 12px 16px; }
  [data-testid="metric-container"] label { color: #888 !important; font-size:.78rem; }

  /* Header banner */
  .f1-banner {
    background: linear-gradient(135deg,#0f0f1c 0%,#1a0530 50%,#050520 100%);
    border: 1px solid #7c4dff44; border-radius: 14px;
    padding: 20px 28px; margin-bottom: 18px;
    display: flex; align-items: center; gap: 18px; }
  .f1-banner h1 { margin:0; font-size:2rem; font-weight:900;
    background:linear-gradient(90deg,#d500f9,#448aff,#00e676);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .f1-banner p { margin:4px 0 0 0; color:#888; font-size:.9rem; }

  /* Status pills */
  .pill { display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:.75rem; font-weight:600; margin:2px; }
  .pill-g { background:#00e67622; color:#00e676; border:1px solid #00e67655; }
  .pill-r { background:#ff174422; color:#ff1744; border:1px solid #ff174455; }
  .pill-y { background:#ffd74022; color:#ffd740; border:1px solid #ffd74055; }
  .pill-b { background:#448aff22; color:#448aff; border:1px solid #448aff55; }

  /* Cards */
  .analysis-card {
    background:#0f0f1c; border:1px solid #181830; border-radius:12px;
    padding:16px 20px; margin:8px 0; }
  .analysis-card h4 { margin:0 0 8px 0; color:#f0f0f0; font-size:1rem; }

  /* Dataframe */
  .dataframe { background:#0f0f1c !important; color:#f0f0f0 !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background:#0f0f1c; border-radius:8px; padding:4px; }
  .stTabs [data-baseweb="tab"] { color:#888; font-weight:600; }
  .stTabs [aria-selected="true"] { color:#f0f0f0 !important; background:#181840 !important; border-radius:6px; }

  /* Buttons */
  .stButton > button {
    background:linear-gradient(135deg,#7c4dff,#448aff);
    color:white; border:none; border-radius:8px; font-weight:700;
    transition: opacity .2s; }
  .stButton > button:hover { opacity:.85; }

  /* Sidebar labels */
  .sidebar-section { color:#ffd740; font-weight:700; font-size:.85rem;
    text-transform:uppercase; letter-spacing:.08em; margin:12px 0 6px 0; }
</style>
""", unsafe_allow_html=True)


# ── Imports (locali) ────────────────────────────────────────────────────
from core.f1_api    import (fetch_event_data, get_available_rounds,
                             get_sc_history, JOLPICA_BASE)
from core.odds_parser import parse_text, parse_image_with_claude, OddsEntry
from core.analysis  import AnalysisEngine
from ml.models      import F1MLEngine
from core.db         import get_db, is_mongo_available
from core.session_tracker import collect_session_info, session_info_to_dict
from core.results    import (fetch_race_result, evaluate_predictions,
                              save_records, load_records, get_evaluated_records,
                              delete_records, get_real_records, history_summary,
                              PredictionRecord)
from ml.evaluator    import (ModelOptimizer, compute_metrics, roi_by_edge_threshold,
                              edge_vs_accuracy_bins)
from components.eval_charts import (reliability_diagram, roi_threshold_chart,
                                     edge_accuracy_chart, outcome_scatter,
                                     metrics_kpi_cards, cs_update_chart)
from components.charts import (edge_chart, prob_comparison_chart, session_heatmap,
                                feature_importance_chart, safety_car_chart,
                                auc_chart, multiples_table, quali_gap_bar)


# ── Session state defaults ───────────────────────────────────────────────
def _init():
    for k,v in [
        ("ml_engine", None),
        ("event_data", None),
        ("entries", []),
        ("preds", []),
        ("multiples", []),
        ("cv", {}),
        ("loaded_event", ""),
        ("pred_records", []),
        ("race_result", None),
        ("optimizer", None),
        ("_session_info", None),
        ("_db", "uninitialized"),
        ("api_log", []),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v
_init()


# ── Cached ML training ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_ml_engine():
    eng = F1MLEngine()
    eng.train()
    return eng


# ── Connessione MongoDB (singleton per sessione) ─────────────────────
if st.session_state.get("_db") == "uninitialized":
    st.session_state["_db"] = get_db()   # None se non configurato → JSON fallback
_db = st.session_state.get("_db")

# ── Telemetria sessione (raccolta una sola volta per sessione) ──────
if st.session_state.get("_session_info") is None:
    try:
        _sinfo = collect_session_info(action="app_open")
        st.session_state["_session_info"] = session_info_to_dict(_sinfo)
    except Exception:
        st.session_state["_session_info"] = {"timestamp": "", "session_hash": ""}

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-section">🏎 Evento F1</div>', unsafe_allow_html=True)

    year = st.selectbox("Anno", [2026, 2025, 2024], index=0)

    @st.cache_data(ttl=3600, show_spinner=False)
    def _rounds(yr):
        return get_available_rounds(yr)

    rounds = _rounds(year)
    if rounds:
        round_labels = {f"R{r['round']} — {r['name']}": r for r in rounds}
        sel_label    = st.selectbox("Round GP", list(round_labels.keys()))
        sel_round    = round_labels[sel_label]
    else:
        st.warning("Calendario non disponibile")
        sel_round = None

    country_name = st.text_input("Country OpenF1",
        value=sel_round.get("country","") if sel_round else "",
        help="Usato per le sessioni FP da OpenF1 (es: Australia, Italy)")

    st.markdown('<div class="sidebar-section">🔑 API Keys (opzionali)</div>', unsafe_allow_html=True)
    anthropic_key = st.text_input("Anthropic API Key", type="password",
        value=os.environ.get("ANTHROPIC_API_KEY",""),
        help="Necessaria solo per il parsing quote da immagine")

    st.markdown('<div class="sidebar-section">⚙️ Impostazioni</div>', unsafe_allow_html=True)
    stake   = st.number_input("Puntata per multipla (€)", 1.0, 50.0, 2.0, 0.5)
    target  = st.number_input("Target vincita multipla (€)", 10.0, 500.0, 25.0, 5.0)
    top_n   = st.slider("Selezioni nel grafico confronto", 5, 15, 10)

    st.markdown("---")
    load_btn = st.button("📡 Carica dati sessioni", width='stretch')

    if load_btn and sel_round:
        with st.spinner("Scaricando dati sessioni reali..."):
            ev = fetch_event_data(
                year=year,
                round_num=int(sel_round["round"]),
                country_name=country_name,
                event_name=sel_round["name"],
            )
            ev.circuit = sel_round.get("circuit", "")
            st.session_state["event_data"]   = ev
            st.session_state["loaded_event"] = sel_round["name"]
            st.session_state["api_log"]      = getattr(ev, "_progress_log", [])
        st.success(f"✅ {len(ev.sessions)} sessioni caricate")

    if st.session_state["api_log"]:
        with st.expander("📋 Log API"):
            for line in st.session_state["api_log"]:
                st.text(line)

    st.markdown("---")
    # Issues manuali
    st.markdown('<div class="sidebar-section">⚠️ Problemi Sessione</div>', unsafe_allow_html=True)
    issues_raw = st.text_area("Driver=valore (0=ok, 1=min, 2=grave)",
        placeholder="Antonelli=2\nVerstappen=2",
        height=80)
    issues_map = {}
    for line in issues_raw.splitlines():
        if '=' in line:
            d,v = line.split('=',1)
            try: issues_map[d.strip()] = int(v.strip())
            except: pass


# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
ev = st.session_state.get("event_data")
event_name = st.session_state.get("loaded_event") or (sel_round["name"] if sel_round else "—")
circuit    = ev.circuit if ev else (sel_round.get("circuit","") if sel_round else "—")

sessions_str = ', '.join(ev.sessions.keys()) if ev else 'nessuna sessione caricata'
st.html(f"""
<div style="background:linear-gradient(135deg,#07070f 0%,#0d0025 40%,#07070f 100%);
            border:1px solid rgba(124,77,255,.35);border-radius:16px;
            padding:22px 32px;margin-bottom:18px;
            display:flex;align-items:center;justify-content:space-between;gap:24px">
  <div style="display:flex;align-items:center;gap:20px">
    <div style="font-size:2.8rem;line-height:1">🎯</div>
    <div>
      <div style="font-size:2rem;font-weight:900;letter-spacing:-.02em;
                  background:linear-gradient(90deg,#d500f9,#7c4dff,#448aff);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  -webkit-text-stroke:0px">BetBreaker</div>
      <div style="color:#888;font-size:.82rem;margin-top:2px;letter-spacing:.04em">
        F1 MACHINE LEARNING ODDS ANALYZER
      </div>
    </div>
  </div>
  <div style="text-align:right">
    <div style="color:#f0f0f0;font-size:.95rem;font-weight:600">{event_name}</div>
    <div style="color:#666;font-size:.78rem;margin-top:3px">{circuit} &nbsp;·&nbsp; {sessions_str}</div>
  </div>
</div>""")


# ── Metriche rapide ──────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Sessioni", len(ev.sessions) if ev else "—")
col2.metric("Piloti (Quali)", len(ev.quali_data) if ev else "—")
col3.metric("Quote caricate", len(st.session_state["entries"]))
col4.metric("Predizioni ML", len(st.session_state["preds"]))
col5.metric("Multipla top", f"€{st.session_state['multiples'][0].win:.2f}" if st.session_state['multiples'] else "—")


# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
tab_about, tab_data, tab_odds, tab_ml, tab_report, tab_eval = st.tabs([
    "🏠 Come Usare",
    "📡 Dati Sessioni",
    "📊 Inserisci Quote",
    "🤖 Analisi ML",
    "📋 Report & Multiple",
    "📈 Valutazione Ex-Post",
])


# ─────────────────────────────────────────────────────────────────────
# TAB 1 — Dati Sessioni
# ─────────────────────────────────────────────────────────────────────
with tab_data:
    if not ev:
        st.info("👈 Seleziona un GP nella sidebar e premi **Carica dati sessioni**")
        st.markdown("""
        <div class="analysis-card">
        <h4>API utilizzate</h4>
        <ul>
        <li><b>Jolpica/Ergast API</b> — qualifiche, sprint race, risultati gara (jolpi.ca/ergast)</li>
        <li><b>OpenF1 API</b> — prove libere FP1/FP2/FP3, tempi giro live (openf1.org)</li>
        </ul>
        Nessuna API key richiesta — completamente pubbliche e gratuite.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Dati sessioni
        st.subheader("📡 Dati recuperati dall'API")
        src_cols = st.columns(len(ev.sessions)+1)
        for i,(sname,src) in enumerate(ev.data_sources.items()):
            src_cols[i].markdown(f'<span class="pill pill-b">{sname}</span><br><small>{src}</small>', unsafe_allow_html=True)
        if ev.errors:
            st.warning("⚠️ " + " · ".join(ev.errors))

        # Heatmap sessioni
        sc_hist = get_sc_history(ev.circuit)
        st.plotly_chart(session_heatmap(ev.sessions, ev.grid), width='stretch', key="pc_1")


        # Gap qualifiche reali
        if ev.quali_data:
            st.plotly_chart(quali_gap_bar(ev.quali_data), width='stretch', key="pc_2")


        # Safety Car
        st.plotly_chart(safety_car_chart(sc_hist, ev.circuit), width='stretch', key="pc_3")


        # Tabella raw
        with st.expander("📋 Dati sessioni grezzi"):
            all_drivers = sorted(ev.grid.keys(), key=lambda d: ev.grid.get(d,99))
            rows = []
            for d in all_drivers:
                row = {"Pilota": d, "Griglia": ev.grid.get(d,"?")}
                for s in ['FP1','FP2','FP3','Sprint','Quali']:
                    row[s] = f"{ev.sessions.get(s,{}).get(d,'-'):.3f}s" if isinstance(ev.sessions.get(s,{}).get(d), float) else "—"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows).set_index("Pilota"), width='stretch')


# ─────────────────────────────────────────────────────────────────────
# TAB 2 — Inserisci Quote
# ─────────────────────────────────────────────────────────────────────
with tab_odds:
    st.subheader("📊 Inserisci le quote del bookmaker")

    mode = st.radio("Modalità input", ["✍️ Testo libero", "📷 Screenshot (Claude Vision)", "🗂️ Esempio demo"],
                    horizontal=True)

    if mode == "✍️ Testo libero":
        st.markdown("""
        **Formati supportati:**
        ```
        # T/T Gara
        Leclerc vs Norris: 1.25 / 3.50
        Russell vs Antonelli: 1.08 / 6.00

        # Migliore Gruppo
        G1: Russell=1.25, Antonelli=4.50, Leclerc=7.00, Verstappen=21.00
        G2: Piastri=2.50, Hamilton=2.85, Hadjar=3.35, Norris=3.75

        # Singoli
        Hamilton Top6: 1.55
        Safety Car: 1.60
        ```
        """)
        odds_text = st.text_area("Incolla le quote qui", height=260, placeholder="Leclerc vs Norris: 1.25 / 3.50\nG1: Russell=1.25, Antonelli=4.50...")
        if st.button("✅ Analizza quote", width='stretch') and odds_text.strip():
            st.session_state["entries"] = parse_text(odds_text)
            st.success(f"✅ {len(st.session_state['entries'])} selezioni caricate")

    elif mode == "📷 Screenshot (Claude Vision)":
        uploaded = st.file_uploader("Carica screenshot Eurobet/Snai/Sisal",
                                     type=["jpg","jpeg","png","webp"])
        if uploaded:
            st.image(uploaded, caption="Screenshot caricato", width='stretch')
            if st.button("🤖 Estrai quote con Claude Vision", width='stretch'):
                if not anthropic_key:
                    st.error("⚠️ Inserisci l'API Key Anthropic nella sidebar")
                else:
                    with st.spinner("Claude sta leggendo lo screenshot..."):
                        mime_map = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","webp":"image/webp"}
                        mime  = mime_map.get(uploaded.name.split('.')[-1].lower(),"image/jpeg")
                        entries = parse_image_with_claude(uploaded.read(), mime, anthropic_key)
                    if entries:
                        st.session_state["entries"] = entries
                        st.success(f"✅ {len(entries)} selezioni estratte dall'immagine")
                    else:
                        st.error("❌ Impossibile estrarre le quote. Controlla la chiave API.")

    else:  # Demo
        DEMO = """# T/T Gara
Leclerc vs Norris: 1.25 / 3.50
Leclerc vs Piastri: 1.35 / 2.85
Norris vs Piastri: 2.65 / 1.40
Leclerc vs Hamilton: 1.30 / 3.00
Norris vs Hamilton: 2.25 / 1.55
Piastri vs Hamilton: 1.83 / 1.83
Russell vs Antonelli: 1.08 / 6.00

# Migliore Gruppo
G1: Russell=1.25, Antonelli=4.50, Leclerc=7.00, Verstappen=21.00
G2: Piastri=2.50, Hamilton=2.85, Hadjar=3.35, Norris=3.75
G3: Lindblad=2.60, Bearman=3.00, Lawson=3.00, Ocon=4.00
G4: Bortoleto=1.90, Hulkenberg=2.35, Gasly=5.00, Albon=7.00"""
        st.code(DEMO, language="")
        if st.button("▶ Carica dati demo (GP Australia 2026 — quote Eurobet reali)", width='stretch'):
            st.session_state["entries"] = parse_text(DEMO)
            st.success(f"✅ {len(st.session_state['entries'])} selezioni demo caricate")

    # Preview
    if st.session_state["entries"]:
        st.markdown("---")
        st.subheader(f"📋 {len(st.session_state['entries'])} selezioni caricate")
        df_e = pd.DataFrame([{
            "Mercato": e.mkt.value,
            "Label": e.label,
            "Quota": e.quota,
            "Gruppo": e.group or "—",
        } for e in st.session_state["entries"]])
        st.dataframe(df_e, width='stretch', height=350)


# ─────────────────────────────────────────────────────────────────────
# TAB 3 — Analisi ML
# ─────────────────────────────────────────────────────────────────────
with tab_ml:
    st.subheader("🤖 Analisi Machine Learning")

    if not st.session_state["entries"]:
        st.info("👈 Prima inserisci le quote nel tab **Inserisci Quote**")
    elif not ev:
        st.warning("⚠️ Nessun dato sessione caricato — l'analisi usa feature di default (griglia parziale)")
        if st.button("🚀 Esegui analisi ML con dati parziali"):
            _run = True
        else:
            _run = False
    else:
        _run = True

    if st.session_state.get("entries") and (ev or not ev):
        run_btn = st.button("🚀 Esegui analisi ML completa", width='stretch',
                             type="primary", key="run_ml")
        if run_btn:
            with st.spinner("⚙️ Training modelli (LR · RF · GBM)..."):
                engine = get_ml_engine()
                st.session_state["ml_engine"] = engine
                st.session_state["cv"]        = engine.cv

            sessions = ev.sessions if ev else {}
            grid     = ev.grid if ev else {}
            sc_hist  = get_sc_history(ev.circuit if ev else "")
            if ev: st.session_state["event_data"] = ev

            with st.spinner("🔮 Calcolo predizioni ed edge..."):
                ae = AnalysisEngine(engine, sessions, grid, issues_map, sc_hist)
                ae.set_groups(st.session_state["entries"])
                preds = ae.predict_all(st.session_state["entries"])
                mults = ae.build_multiples(preds, stake=stake, target=target)
                st.session_state["preds"]     = preds
                st.session_state["multiples"] = mults

            st.success(f"✅ {len(preds)} predizioni — {len([p for p in preds if p.edge>0.04])} 🟢 — {len([p for p in preds if abs(p.edge)<=0.04])} 🟡 — {len([p for p in preds if p.edge<-0.05])} 🔴")

    preds = st.session_state["preds"]
    if preds:
        st.markdown("---")
        # KPI row
        top3 = sorted(preds, key=lambda p: p.edge, reverse=True)[:3]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🟢 Sottostimate", len([p for p in preds if p.edge>0.04]))
        c2.metric("🟡 Eque", len([p for p in preds if -0.04<=p.edge<=0.04]))
        c3.metric("🔴 Sovrastimate", len([p for p in preds if p.edge<-0.05]))
        c4.metric("Best edge", f"{max(p.edge_pct for p in preds):+.1f}%" if preds else "—")

        st.plotly_chart(edge_chart(preds), width='stretch', key="pc_4")


        c_left, c_right = st.columns([2,1])
        with c_left:
            st.plotly_chart(prob_comparison_chart(preds, top_n), width='stretch', key="pc_5")

        with c_right:
            st.plotly_chart(auc_chart(st.session_state["cv"]), width='stretch', key="pc_6")

            _eng = st.session_state.get("ml_engine")
            fi = _eng.feature_importance() if _eng else None
            if fi is not None:
                st.plotly_chart(feature_importance_chart(fi), width='stretch', key="pc_7")



# ─────────────────────────────────────────────────────────────────────
# TAB 4 — Report & Multiple
# ─────────────────────────────────────────────────────────────────────
with tab_report:
    st.subheader("📋 Report completo")

    preds = st.session_state["preds"]
    mults = st.session_state["multiples"]
    ev    = st.session_state.get("event_data")

    if not preds:
        st.info("👈 Prima esegui l'analisi ML nel tab **Analisi ML**")
    else:
        # ── Riepilogo top picks ──────────────────────────────────
        st.markdown("### 🟢 Selezioni più sottostimate")
        top_picks = sorted(preds, key=lambda p: p.edge, reverse=True)[:10]
        for i,p in enumerate(top_picks):
            pill_cls = "pill-g" if p.edge>0.04 else "pill-y" if p.edge>-0.05 else "pill-r"
            st.markdown(
                f'<div class="analysis-card">'
                f'<b>{i+1}. {p.label}</b> &nbsp; '
                f'<span class="pill {pill_cls}">{p.verdict}</span><br>'
                f'Quota: <b>{p.quota:.2f}</b> &nbsp;→&nbsp; Fair Quota: <b>{p.fair_q:.2f}</b> &nbsp;&nbsp;'
                f'Implicita: {p.impl*100:.1f}% &nbsp; Ensemble: {p.ens*100:.1f}% &nbsp; '
                f'<b style="color:{p.color}">Edge: {p.edge_pct:+.1f}%</b>'
                f'</div>', unsafe_allow_html=True)

        st.markdown("---")
        # ── Multiple ────────────────────────────────────────────
        st.markdown("### 🎰 5 Multiple ML-Validated")
        st.plotly_chart(multiples_table(mults), width='stretch', key="pc_8")


        for i,m in enumerate(mults):
            risk  = ["⭐⭐","⭐⭐","⭐⭐⭐","⭐⭐⭐","⭐⭐⭐⭐⭐"][i]
            color = "#00e676" if m.win>=25 else "#ffd740" if m.win>=12 else "#ff6d00"
            with st.expander(f"{m.label}  |  Quota comb: {m.q_comb:.2f}x  |  Vincita: €{m.win:.2f}  {risk}"):
                for p in m.picks:
                    st.markdown(f"- **{p.label}** — Q={p.quota:.2f} &nbsp; Edge <b style='color:{p.color}'>{p.edge_pct:+.1f}%</b>",
                                unsafe_allow_html=True)
                st.markdown(f"_{m.note}_")
                st.markdown(f"<h3 style='color:{color}'>💰 Vincita: €{m.win:.2f}</h3>", unsafe_allow_html=True)

        st.markdown("---")
        # ── Tabella completa ────────────────────────────────────
        st.markdown("### 📊 Tabella completa predizioni")
        df = pd.DataFrame([{
            "Mercato":    p.entry.mkt.value,
            "Selezione":  p.label,
            "Quota":      p.quota,
            "Fair Q":     round(p.fair_q, 2),
            "Implicita %":f"{p.impl*100:.1f}",
            "Ensemble %": f"{p.ens*100:.1f}",
            "Edge %":     f"{p.edge_pct:+.1f}",
            "Verdetto":   p.verdict,
        } for p in sorted(preds, key=lambda x: x.edge, reverse=True)])
        st.dataframe(df, width='stretch', height=500)

        # Download CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Scarica CSV", csv, "f1_predictions.csv", "text/csv")

        # Grafici sessioni in fondo
        if ev:
            st.markdown("---")
            st.markdown("### 📡 Dati sessioni reali")
            sc_hist = get_sc_history(ev.circuit)
            sc_pred = next((p for p in preds if p.entry.mkt.value=="Safety Car"),None)
            c1,c2 = st.columns(2)
            with c1: st.plotly_chart(session_heatmap(ev.sessions, ev.grid), width='stretch', key="pc_9")

            with c2: st.plotly_chart(safety_car_chart(sc_hist, ev.circuit, sc_pred), width='stretch', key="pc_10")



# ─────────────────────────────────────────────────────────────────────
# TAB 5 — About
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# TAB 5 — VALUTAZIONE EX-POST & OTTIMIZZAZIONE
# ─────────────────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("📈 Valutazione ex-post & Ottimizzazione modello")

    from ml.models import DEFAULT_CS

    # ── Sidebar di questo tab: carica risultato gara ─────────────
    st.markdown("""
    <div style="background:#0f0f1c;border:1px solid #181830;border-radius:12px;padding:16px 20px;margin-bottom:16px">
    <b>Come funziona il ciclo di feedback:</b><br>
    <ol style="margin:8px 0;padding-left:20px;color:#ccc">
    <li>Esegui l'analisi ML <i>prima</i> della gara → le predizioni vengono salvate</li>
    <li>Dopo la gara, carica il risultato ufficiale dall'API</li>
    <li>Il sistema confronta le predizioni con i risultati reali</li>
    <li>Le metriche ex-post guidano l'ottimizzazione del modello</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    col_load, col_status = st.columns([2,1])
    with col_load:
        _default_year  = int(sel_round["year"])  if sel_round and "year"  in sel_round else 2026
        _default_round = int(sel_round["round"]) if sel_round and "round" in sel_round else 1
        eval_year  = st.number_input("Anno",  2024, 2026, _default_year,  key="eval_year")
        eval_round = st.number_input("Round", 1,    24,   _default_round, key="eval_round")
    # Garanzia: sempre int validi anche se il widget non ha ancora restituito un valore
    eval_year  = int(eval_year  or _default_year)
    eval_round = int(eval_round or _default_round)
    with col_status:
        st.markdown("<br>", unsafe_allow_html=True)
        load_result_btn = st.button("📡 Carica risultato gara", width='stretch')

    if load_result_btn:
        with st.spinner("Scaricando risultato gara da Jolpica..."):
            rr = fetch_race_result(int(eval_year), int(eval_round))
        if rr and rr.results:
            st.session_state["race_result"] = rr
            st.success(f"✅ {rr.event_name} — {len(rr.results)} piloti · {rr.date}")
        else:
            st.warning("⚠️ Risultato non ancora disponibile per questo round. "
                       "Puoi inserire i risultati manualmente qui sotto.")

    # ── Input manuale risultati ──────────────────────────────────
    with st.expander("✏️ Inserisci risultati manualmente (se API non disponibile)"):
        st.markdown("**Formato:** `Pilota=posizione` (es: `Russell=1, Norris=2, Leclerc=3`)")
        manual_results = st.text_area("Risultati gara", height=100,
            placeholder="Russell=1\nNorris=2\nLeclerc=3\nHamilton=4\nPiastri=5\nAntonelli=6")
        if st.button("✅ Applica risultati manuali") and manual_results.strip():
            from core.results import RaceResult, DriverResult
            rr_manual = RaceResult(
                year=int(eval_year), round_num=int(eval_round),
                event_name=f"Round {eval_round} {eval_year} (manuale)",
                circuit="", date=""
            )
            for line in manual_results.strip().splitlines():
                if "=" in line:
                    d, pos = line.split("=", 1)
                    d = d.strip(); pos = int(pos.strip())
                    rr_manual.results[d] = DriverResult(d, pos, pos, 0.0, "Finished")
            # Build H2H outcomes
            items = list(rr_manual.results.items())
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    d1,r1 = items[i]; d2,r2 = items[j]
                    key = f"{d1} vs {d2}"
                    rr_manual.h2h_outcomes[key] = r1.position < r2.position
            st.session_state["race_result"] = rr_manual
            st.success(f"✅ {len(rr_manual.results)} risultati inseriti")

    rr = st.session_state.get("race_result")
    preds = st.session_state.get("preds", [])

    # ── Mostra risultato gara ────────────────────────────────────
    if rr and rr.results:
        st.markdown("---")
        st.markdown(f"### 🏁 Risultato: {rr.event_name}")
        top10 = sorted(rr.results.values(), key=lambda r: r.position)[:10]
        res_cols = st.columns(5)
        for i, dr in enumerate(top10[:5]):
            medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][i]
            res_cols[i].metric(f"{medal} P{dr.position}", dr.driver, dr.status)
        res_cols2 = st.columns(5)
        for i, dr in enumerate(top10[5:10]):
            res_cols2[i].metric(f"P{dr.position}", dr.driver, dr.status)

    # ── Converti predizioni → PredictionRecord e valuta ─────────
    if preds and rr:
        st.markdown("---")
        if st.button("🔄 Valuta predizioni con risultato reale", width='stretch', type="primary"):
            ev_name = rr.event_name
            raw_records = []
            for p in preds:
                raw_records.append(PredictionRecord(
                    year=int(eval_year), round_num=int(eval_round),
                    event_name=ev_name,
                    market=p.entry.mkt.value,
                    label=p.label,
                    driver1=p.entry.driver1,
                    driver2=p.entry.driver2,
                    quota=p.quota,
                    implied_prob=p.impl,
                    ensemble_prob=p.ens,
                    edge=p.edge,
                ))
            evaluated = evaluate_predictions(raw_records, rr)
            # Marca esplicitamente come dati reali + arricchisci con telemetria
            _si = st.session_state.get("_session_info", {})
            try:
                _sinfo_now = collect_session_info(
                    action="evaluate",
                    gp_year=int(eval_year),
                    gp_round=int(eval_round),
                    do_geoip=False,  # veloce — no lookup esterno
                )
            except Exception:
                _sinfo_now = None
            for r in evaluated:
                r.source = "real"
                if _sinfo_now:
                    r.timestamp      = _sinfo_now.timestamp
                    r.session_hash   = _sinfo_now.session_hash
                    r.ip_hash        = _sinfo_now.ip_hash
                    r.ip_country     = _sinfo_now.ip_country
                    r.browser_family = _sinfo_now.browser_family
                    r.os_family      = _sinfo_now.os_family
                    r.is_mobile      = _sinfo_now.is_mobile
                    r.language       = _sinfo_now.language
                    r.action         = "evaluate"
            n_saved = save_records(evaluated, db=_db, year=int(eval_year))
            st.session_state["pred_records"] = load_records()
            n_eval = len([r for r in evaluated if r.outcome is not None])
            st.success(f"✅ {n_eval}/{len(evaluated)} predizioni valutate · {n_saved} nuovi record salvati")

    # ── Carica storico ───────────────────────────────────────────
    all_records = st.session_state.get("pred_records") or (load_records(db=_db, year=eval_year) if eval_year else [])
    if not all_records:
        all_records = []
    st.session_state["pred_records"] = all_records

    # Solo record reali dell'anno selezionato per il modello
    # get_real_records already filters — re-derive from all_records for display
    real_records      = [r for r in all_records if getattr(r, 'source', 'real') == 'real']
    evaluated_records = [r for r in real_records if r.outcome is not None]

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("📦 Storico reale", len(real_records), f"Anno {int(eval_year)}")
    sc2.metric("✅ Valutate", len(evaluated_records))
    sc3.metric("⏳ In attesa", len(real_records) - len(evaluated_records))

    st.markdown("---")

    if len(evaluated_records) < 3:
        st.info("ℹ️ Servono almeno 3 predizioni valutate per visualizzare le metriche. "
                "Carica più risultati di gara e premi 'Valuta predizioni'.")

        # Dati demo per visualizzazione
        if st.button("🎲 Carica dati demo (30 predizioni simulate)"):
            import random
            random.seed(42)
            demo_recs = []
            drivers_pool = ["Russell","Leclerc","Hamilton","Piastri","Norris","Antonelli","Hadjar","Verstappen"]
            markets = ["T/T Gara","Migliore Gruppo","Top 6"]
            for i in range(30):
                d1,d2 = random.sample(drivers_pool,2)
                prob = random.uniform(0.30, 0.75)
                impl = random.uniform(0.28, 0.72)
                edge = prob - impl
                q    = 1/impl
                outcome = random.random() < (prob * 0.85 + random.uniform(-0.1,0.1))
                demo_recs.append(PredictionRecord(
                    year=2026, round_num=random.randint(1,5),
                    event_name=f"Demo GP {random.randint(1,5)}",
                    market=random.choice(markets),
                    label=f"{d1} batte {d2}",
                    driver1=d1, driver2=d2,
                    quota=round(q,2), implied_prob=round(impl,4),
                    ensemble_prob=round(prob,4), edge=round(edge,4),
                    outcome=outcome,
                    outcome_position_d1=random.randint(1,10),
                    outcome_position_d2=random.randint(1,10),
                ))
            # Tag esplicitamente come demo → NON inquinano le metriche reali
            for r in demo_recs:
                r.source = "demo"
            save_records(demo_recs, db=_db, year=int(eval_year))
            st.session_state["pred_records"] = load_records(db=_db, year=int(eval_year))
            st.rerun()

    else:
        # ══════════════════════════════════════════════════════
        # SEZIONE METRICHE
        # ══════════════════════════════════════════════════════
        metrics = compute_metrics(evaluated_records)

        if metrics:
            st.markdown("### 📊 Metriche ex-post")
            st.html(metrics_kpi_cards(metrics))

            # Edge accuracy per fascia
            ea = metrics.edge_accuracy
            ea_cols = st.columns(3)
            for i, (fascia, acc) in enumerate(ea.items()):
                if not np.isnan(acc):
                    col = "#00e676" if "Verde" in fascia else "#ffd740" if "Giallo" in fascia else "#ff1744"
                    ea_cols[i].html(
                        f'<div style="background:#0f0f1c;border:1px solid #181830;border-radius:10px;'
                        f'padding:12px;text-align:center;border-top:3px solid {col}">'
                        f'<div style="color:#888;font-size:.72rem">{fascia}</div>'
                        f'<div style="color:{col};font-size:1.5rem;font-weight:900">{acc*100:.1f}%</div>'
                        f'<div style="color:#666;font-size:.7rem">win rate osservato</div></div>'
                    )

        st.markdown("---")

        # ── 4 grafici principali ─────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(reliability_diagram(evaluated_records), width='stretch', key="pc_11")

        with c2:
            st.plotly_chart(outcome_scatter(evaluated_records), width='stretch', key="pc_12")


        st.plotly_chart(roi_threshold_chart(evaluated_records), width='stretch', key="pc_13")

        st.plotly_chart(edge_accuracy_chart(evaluated_records), width='stretch', key="pc_14")


        # ── Tabella errori per mercato ──────────────────────
        st.markdown("### 🔍 Errori sistematici per mercato")
        opt = ModelOptimizer(evaluated_records)
        se  = opt.systematic_errors()
        if not se.empty:
            st.dataframe(se.style.background_gradient(
                subset=["ROI %"], cmap="RdYlGn"), width='stretch')

        # ══════════════════════════════════════════════════════
        # SEZIONE OTTIMIZZAZIONE
        # ══════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### ⚙️ Ottimizzazione del modello")

        # Raccomandazioni
        recs = opt.optimization_recommendations()
        for r in recs:
            col = "#00e676" if "ALTA" in r["priorità"] and "POSITIVO" in r["priorità"]                   else "#ff1744" if "ALTA" in r["priorità"]                   else "#ffd740"
            st.markdown(
                f'<div style="background:#0f0f1c;border-left:4px solid {col};'
                f'border-radius:0 10px 10px 0;padding:12px 16px;margin:8px 0">'
                f'<b style="color:{col}">{r["priorità"]}</b><br>'
                f'<span style="color:#ccc">{r["raccomandazione"]}</span><br>'
                f'<code style="color:#888;font-size:.8rem">→ {r["azione"]}</code>'
                f'</div>', unsafe_allow_html=True)

        # ── Calibrazione isotonica ───────────────────────────
        st.markdown("#### 🎯 Calibrazione isotonica")
        calib_info = """La calibrazione isotonica mappa le probabilità predette dal modello
        verso le frequenze reali osservate. Riduce il Brier Score quando il modello è
        sistematicamente ottimista o pessimista."""
        st.info(calib_info)

        if len(evaluated_records) >= 20:
            if st.button("🔧 Calibra modello (isotonic regression)", width='stretch'):
                opt.fit_calibration()
                imp = opt.calibration_improvement()
                st.session_state["optimizer"] = opt
                st.success(f"✅ Calibrazione applicata! "
                           f"Brier: {imp['brier_raw']:.4f} → {imp['brier_calibrated']:.4f} "
                           f"(miglioramento: {imp['improvement']:.4f})")
        else:
            st.warning(f"Servono 20+ predizioni valutate per la calibrazione "
                       f"(hai {len(evaluated_records)})")

        # ── Constructor Strength empirica ────────────────────
        st.markdown("#### 🏎 Constructor Strength empirica")
        cs_est = opt.estimate_constructor_strength()
        if cs_est:
            st.plotly_chart(cs_update_chart(cs_est, DEFAULT_CS), width='stretch', key="pc_15")

            # Mostra codice aggiornato
            lines = ["DEFAULT_CS = {"]
            for d, v in sorted(cs_est.items(), key=lambda x:-x[1]):
                lines.append(f"    '{d}': {v:.3f},")
            lines.append("}")
            cs_code = "\n".join(lines)
            with st.expander("📋 Codice aggiornato per ml/models.py"):
                st.code(cs_code, language="python")
                st.caption("Copia questo dizionario in ml/models.py → DEFAULT_CS per aggiornare il modello")
        else:
            st.info("Servono almeno 10 predizioni H2H valutate per stimare la forza costruttori")

        # ── ROI per soglia — tabella numerica ────────────────
        with st.expander("📊 Tabella ROI per soglia edge"):
            df_roi = roi_by_edge_threshold(evaluated_records)
            if not df_roi.empty:
                best_thr = df_roi.loc[df_roi['roi_pct'].idxmax()]
                st.success(f"Soglia ottimale: **{best_thr['edge_threshold']:.0f}%** "
                           f"→ ROI {best_thr['roi_pct']:+.1f}% su {best_thr['n_bets']:.0f} scommesse")
                st.dataframe(df_roi, width='stretch')

        # ── Export storico ───────────────────────────────────
        st.markdown("---")
        csv_h = pd.DataFrame([{
            "Anno":r.year,"Round":r.round_num,"Evento":r.event_name,
            "Mercato":r.market,"Selezione":r.label,
            "Quota":r.quota,"Impl%":round(r.implied_prob*100,1),
            "ML%":round(r.ensemble_prob*100,1),"Edge%":round(r.edge*100,1),
            "Outcome": "✅" if r.outcome else "❌" if r.outcome is not None else "—",
        } for r in all_records]).to_csv(index=False).encode()
        st.download_button("⬇️ Scarica storico completo CSV",
                           csv_h, "f1_predictions_history.csv", "text/csv", width='stretch')


with tab_about:

    # ── HERO ──────────────────────────────────────────────────────────
    st.html("""
<div style="background:linear-gradient(135deg,#0a0020,#07070f,#001025);
            border:1px solid rgba(68,138,255,.25);border-radius:16px;
            padding:32px 36px;margin-bottom:28px;text-align:center">
  <div style="font-size:3rem;margin-bottom:8px">🎯</div>
  <div style="font-size:1.6rem;font-weight:900;
              background:linear-gradient(90deg,#d500f9,#7c4dff,#448aff,#00e676);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    Benvenuto su BetBreaker
  </div>
  <div style="color:#888;font-size:.95rem;margin-top:8px;max-width:600px;margin-left:auto;margin-right:auto">
    Piattaforma di analisi ML per le quote Formula 1.<br>
    Dati reali di sessione · Ensemble di 3 modelli · Edge calcolato sul margine del bookmaker.
  </div>
</div>""")

    # ── GUIDA PASSO-PASSO ─────────────────────────────────────────────
    st.markdown("## 📋 Guida all'utilizzo")

    steps = [
        ("1", "📡", "Seleziona e carica i dati di sessione",
         """Nella **sidebar a sinistra**:
- Scegli l'**anno** e il **round GP** dal menu a tendina (il calendario viene scaricato automaticamente)
- Verifica il campo **Country OpenF1** (serve per le prove libere): es. *Australia*, *Italy*, *Monaco*
- Premi **"📡 Carica dati sessioni"**

L'app scarica in automatico:
- **FP1 / FP2 / FP3** → da [OpenF1 API](https://openf1.org) *(tempi giro reali, disponibile dal 2023)*
- **Qualifiche** → da [Jolpica/Ergast](https://jolpi.ca/ergast) *(tempi Q1/Q2/Q3, posizioni)*
- **Sprint Race** → da [Jolpica/Ergast](https://jolpi.ca/ergast) *(se il round la prevede)*

> 💡 Nessuna API key necessaria per i dati di sessione.""",
         "#448aff"),

        ("2", "📊", "Inserisci le quote del bookmaker",
         """Vai al tab **"📊 Inserisci Quote"** e scegli una delle tre modalità:

**✍️ Testo libero** — incolla le quote in formato semplice:
```
Leclerc vs Norris: 1.25 / 3.50
G1: Russell=1.25, Antonelli=4.50, Leclerc=7.00
Hamilton Top6: 1.55
Safety Car: 1.60
```

**📷 Screenshot (Claude Vision)** — carica uno screenshot dell'app Eurobet / Snai / Sisal.
Claude estrae automaticamente tutte le quote.
*Richiede API Key Anthropic (gratuita su [console.anthropic.com](https://console.anthropic.com))*

**🗂️ Esempio demo** — carica le quote reali del GP Australia 2026 per esplorare l'app.

Mercati supportati: T/T Gara · T/T 1° Giro · Migliore Gruppo · Top 6 · Podio · Safety Car""",
         "#7c4dff"),

        ("3", "🤖", "Esegui l'analisi ML",
         """Vai al tab **"🤖 Analisi ML"** e premi **"🚀 Esegui analisi ML completa"**.

L'engine:
1. Addestra i 3 modelli (LR + RF + GBM) — ~2 secondi
2. Costruisce le 11 feature per ogni pilota usando i dati di sessione reali
3. Calcola la probabilità ensemble per ogni selezione
4. Confronta con la probabilità implicita del bookmaker (corretta per margine)
5. Calcola l'**edge** e classifica ogni selezione

**Leggi i risultati:**
- 🟢 **Edge > +4%** → Quota sottostimata, potenziale valore
- 🟡 **Edge ±4%** → Quota equa, skip
- 🔴 **Edge < -5%** → Quota sovrastimata, evitare

> ⚠️ Se i dati di sessione non sono ancora disponibili (GP in corso), l'analisi usa i valori di default basati sulla griglia di partenza.""",
         "#00e676"),

        ("4", "📋", "Esamina il report e le multiple",
         """Vai al tab **"📋 Report & Multiple"**:

- **Tabella predizioni** — tutte le selezioni ordinate per edge decrescente, con quota fair calcolata
- **5 Multiple ML-validated** — costruite automaticamente ottimizzando edge e target vincita
- **Download CSV** — esporta tutte le predizioni per analisi esterne
- Puoi modificare **puntata** e **target vincita** dalla sidebar prima di eseguire l'analisi""",
         "#ffd740"),

        ("5", "📈", "Valuta ex-post e ottimizza il modello",
         """Vai al tab **"📈 Valutazione Ex-Post"** dopo la gara:

1. Seleziona anno e round e premi **"📡 Carica risultato gara"** → i risultati ufficiali vengono scaricati da Jolpica
2. Oppure inserisci manualmente i risultati (es. *Russell=1, Norris=2...*)
3. Premi **"🔄 Valuta predizioni"** → l'app confronta ogni predizione col risultato reale e salva lo storico in `data/history.json`

Con abbastanza dati accumulati (20+ predizioni) puoi:
- Applicare la **calibrazione isotonica** per correggere il bias del modello
- Trovare la **soglia edge ottimale** per massimizzare il ROI
- Aggiornare la **forza costruttori** con stime empiriche dai risultati reali

> 💾 Lo storico delle predizioni viene salvato su disco e persiste tra i riavvii dell'app.""",
         "#ff6d00"),
    ]

    for num, icon, title, body, color in steps:
        with st.expander(f"{icon}  Step {num} — {title}", expanded=(num=="1")):
            st.markdown(body)

    st.markdown("---")

    # ── FAQ ────────────────────────────────────────────────────────────
    st.markdown("## ❓ Domande frequenti")

    faqs = [
        ("A cosa serve l'API Key Anthropic?",
         "È **opzionale** e serve **solo** per la modalità screenshot: carica un'immagine dell'app del bookmaker e Claude Vision estrae automaticamente tutte le quote senza che tu debba digitarle. Se inserisci le quote a mano (formato testo), non ti serve. Puoi ottenerne una gratuita su [console.anthropic.com](https://console.anthropic.com)."),
        ("Da dove vengono scaricati i dati?",
         "Da due API pubbliche e gratuite:\n- **[OpenF1](https://openf1.org)** per FP1/FP2/FP3 — dati disponibili dal 2023\n- **[Jolpica/Ergast](https://jolpi.ca/ergast)** per Qualifiche, Sprint, Calendario — dati dal 1950\n\nNessuna registrazione o API key richiesta. I dati vengono scaricati live ogni volta che premi 'Carica dati sessioni'."),
        ("Il modello ML persiste tra un riavvio e l'altro?",
         "**Parzialmente.** Il training ML (LR/RF/GBM) viene rieseguito ad ogni avvio — richiede ~2 secondi ed è deterministico. Lo **storico delle predizioni valutate** invece viene salvato su disco in `data/history.json` e persiste tra i riavvii: metriche, calibrazione e ottimizzazioni si accumulano gara dopo gara."),
        ("Posso usare l'app per gare non-F1?",
         "L'architettura delle feature e il dataset sintetico di training sono calibrati su Formula 1. I mercati T/T e Migliore Gruppo esistono però anche in MotoGP e altri sport a classifica. Con le dovute modifiche a `DEFAULT_CS` e al training set, potrebbe essere adattata."),
        ("Quanto sono affidabili le predizioni?",
         "L'AUC-ROC sul dataset sintetico è ~0.71 (vs baseline 0.50). L'affidabilità reale dipende molto dalla qualità dei dati di sessione e dalla stabilità del fine-settimana. L'app è uno strumento di analisi statistica, non un oracolo. Monitora sempre le metriche ex-post nella tab Valutazione."),
    ]

    for q, a in faqs:
        with st.expander(f"**{q}**"):
            st.markdown(a)

    st.markdown("---")

    # ── SEZIONE MODELLO ML ─────────────────────────────────────────────
    st.markdown("## 🤖 Il modello ML")

    col_arch, col_feat = st.columns([1, 1])

    with col_arch:
        st.markdown("""
### Architettura Ensemble

L'engine utilizza tre modelli combinati in un ensemble pesato:

| Modello | Parametri chiave | Peso ensemble |
|---------|-----------------|---------------|
| **Logistic Regression** | C=0.5 | 25% |
| **Random Forest** | 400 alberi, depth=9 | 38% |
| **Gradient Boosting** | 300 step, lr=0.04 | 37% |

I pesi sono stati ottimizzati minimizzando il Brier Score sul dataset sintetico di validazione.
L'ensemble cattura sia relazioni lineari (LR) che non-linearità complesse (RF/GBM).

**Training:** ~6.000 gare sintetiche generate con distribuzione calibrata su F1 2018–2025.
**Validazione:** cross-validation 5-fold, metrica AUC-ROC (~0.71).
""")

    with col_feat:
        st.markdown("""
### Feature Engineering (11 variabili)

Per ogni match H2H, il vettore input è la **differenza** delle feature tra i due piloti:

| Feature | Fonte dati | Stabilità cross-era |
|---------|-----------|---------------------|
| `grid_pos` | Qualifiche API | 🟡 Media |
| `quali_gap` | Qualifiche API | 🟡 Media |
| `avg_fp_gap` | FP1+FP2+FP3 API | 🟡 Media |
| `best_fp_gap` | FP1+FP2+FP3 API | 🟢 Alta |
| `fp_consistency` | FP1+FP2+FP3 API | 🟢 Alta |
| `trend_improvement` | FP1 vs FP3 API | 🟢 Alta |
| `constructor_strength` | Config | 🔴 Bassa |
| `driver_experience` | Config | 🟢 Alta |
| `session_issues` | Input manuale | 🟡 Media |
| `fp2_gap` | FP2 API | 🟡 Media |
| `sprint_gap` | Sprint API | 🟡 Media |
""")

    st.markdown("""
### Calcolo Edge

```
Prob. implicita corretta (H2H) = (1 / Q_D1) / (1/Q_D1 + 1/Q_D2)

Edge = Prob. ML Ensemble  −  Prob. implicita corretta

🟢  Edge > +4%   →  Quota SOTTOSTIMATA dal bookmaker  →  potenziale valore
🟡  Edge ±4%    →  Quota EQUA  →  nessun vantaggio statistico
🔴  Edge < −5%  →  Quota SOVRASTIMATA dal bookmaker  →  evitare
```

La correzione per margine è fondamentale: un bookmaker con margine del 6% porta le quote
implicite a sommare ~106% invece di 100%. Senza correzione si sovrastima sistematicamente
il valore delle quote.

### Limiti del modello

- Il training è su dati **sintetici**: le predizioni migliorano con l'accumulo di dati reali nella tab Valutazione
- La `constructor_strength` è **hardcoded**: aggiornala manualmente dopo ogni cambio significativo di rendimento
- Il modello non conosce **meteo, incidenti in gara, strategie gomme**: usa il campo "Problemi sessione" nella sidebar per segnalare anomalie
""")

    st.markdown("---")
    st.markdown("""
> ⚠️ **Disclaimer**: BetBreaker è sviluppato esclusivamente a scopo informativo e di ricerca statistica.
> Non costituisce un consiglio di scommessa. Le predizioni ML sono indicatori probabilistici, non certezze.
> **Gioca responsabilmente.** Numero Verde: **800 921 121** · [giocaresponsabile.it](https://www.giocaresponsabile.it)
""")


# ── Footer ────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#444; font-size:.75rem; margin-top:32px; padding:16px 0; border-top:1px solid #181830;">
  🎯 BetBreaker &nbsp;·&nbsp; Dati: OpenF1 API + Jolpica/Ergast API &nbsp;·&nbsp; ML: LR · RF · GBM &nbsp;·&nbsp; ⚠️ Gioca responsabilmente
</div>
""", unsafe_allow_html=True)