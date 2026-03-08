# 🏎 F1 ML Odds Analyzer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

**Web app ML per analisi quote F1 con dati reali da API pubbliche.**

Inserisci le quote del bookmaker (testo o screenshot), carica i tempi reali di FP1/FP2/FP3/Qualifiche/Sprint via API, e ricevi predizioni ML con edge corretto per margine e multiple ottimizzate.

[Installazione](#-installazione) · [Come funziona](#-come-funziona) · [API](#-api-dati-reali) · [Modello ML](#-modello-ml) · [Formato quote](#-formato-quote-testo)

</div>

---

## ✨ Funzionalità

| Feature | Descrizione |
|---------|-------------|
| 📡 **Dati reali** | FP1/FP2/FP3 da OpenF1 API · Qualifiche/Sprint da Jolpica · Zero API key |
| 📷 **Quote da screenshot** | Claude Vision estrae quote da immagini Eurobet/Snai/Sisal |
| ✍️ **Quote da testo** | Formato libero: T/T, Migliore Gruppo, Top6, Podio, Safety Car |
| 🤖 **Ensemble ML** | LR + Random Forest + Gradient Boosting su 11 feature di sessione |
| 📊 **Edge analysis** | Confronto prob. ML vs implicita bookmaker corretta per margine |
| 🎰 **Multiple builder** | 5 multiple ottimizzate automaticamente per edge e target vincita |
| 📋 **Report completo** | Tabella ordinata per edge + download CSV |
| 🌙 **Dark UI** | Tema racing-inspired con grafici Plotly interattivi |

---

## 🚀 Installazione

### Prerequisiti
- Python 3.10+

```bash
# Clona il repo
git clone https://github.com/TUO_USERNAME/f1-ml-odds-analyzer.git
cd f1-ml-odds-analyzer

# Installa dipendenze
pip install -r requirements.txt

# (Opzionale) API key Anthropic per parsing screenshot
export ANTHROPIC_API_KEY="sk-ant-..."

# Avvia
streamlit run app.py
```

Apri **http://localhost:8501** 🎉

### Con virtual environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎮 Flusso di lavoro

```
1.  Sidebar     →  Seleziona anno + round GP
2.  Sidebar     →  Premi "Carica dati sessioni"
                   └─ FP1/FP2/FP3  via OpenF1 API
                   └─ Qualifiche   via Jolpica/Ergast API
                   └─ Sprint Race  via Jolpica/Ergast API (se disponibile)

3.  Tab Quote   →  Incolla quote testo  OPPURE  carica screenshot
4.  Tab ML      →  Premi "Esegui analisi ML completa"
5.  Tab Report  →  Esamina edge + multiple + scarica CSV
```

---

## 📐 Struttura del progetto

```
f1-ml-odds-analyzer/
│
├── app.py                      ← Entry point Streamlit (5 tab)
├── requirements.txt
├── .streamlit/
│   └── config.toml             ← Tema dark (viola, verde neon)
│
├── core/
│   ├── f1_api.py               ← Client OpenF1 + Jolpica/Ergast
│   ├── odds_parser.py          ← Parser quote: testo + Claude Vision
│   └── analysis.py             ← Motore: ML + edge + multiple builder
│
├── ml/
│   └── models.py               ← F1MLEngine: LR + RF + GBM training
│
└── components/
    └── charts.py               ← Grafici Plotly dark-theme
```

---

## 📡 API Dati Reali

### OpenF1 API — Prove Libere
> 🔗 [openf1.org](https://openf1.org) · Gratuita · Nessuna API key · Dati dal 2023

Recupera i tempi giro di ogni pilota per le sessioni FP1, FP2, FP3.

```
GET https://api.openf1.org/v1/sessions?year=2026&country_name=Australia
GET https://api.openf1.org/v1/laps?session_key=<key>
GET https://api.openf1.org/v1/drivers?session_key=<key>
```

Il gap dal leader viene calcolato come:
```python
best_lap_per_driver = {driver: min(valid_laps)}   # scarta giri < 60s
gap = driver_best_lap - min(best_lap_per_driver.values())
gap = min(gap, 5.0)   # cappa a 5s per outlier
```

### Jolpica / Ergast API — Qualifiche & Sprint
> 🔗 [jolpi.ca/ergast](https://api.jolpi.ca/ergast/f1) · Gratuita · Nessuna API key · Dati dal 1950

```
GET https://api.jolpi.ca/ergast/f1/2026.json                        ← Calendario
GET https://api.jolpi.ca/ergast/f1/2026/1/qualifying.json           ← Qualifiche
GET https://api.jolpi.ca/ergast/f1/2026/1/sprint.json               ← Sprint Race
```

I dati vengono popolati man mano che le sessioni si svolgono nel weekend.

---

## 🤖 Modello ML

### Feature Engineering (11 feature per pilota)

| # | Feature | Fonte | Descrizione |
|---|---------|-------|-------------|
| 1 | `grid_pos` | Qualifiche API | Posizione di partenza |
| 2 | `quali_gap` | Qualifiche API | Gap dal poleman in Q3 (s) |
| 3 | `avg_fp_gap` | FP1+FP2+FP3 | Gap medio prove libere (s) |
| 4 | `best_fp_gap` | FP1+FP2+FP3 | Miglior gap in FP (s) |
| 5 | `fp_consistency` | FP1+FP2+FP3 | Deviazione std gap FP |
| 6 | `trend_improvement` | FP1 vs FP3 | Miglioramento FP1→FP3 (s) |
| 7 | `constructor_strength` | Config | Forza costruttore (0–1) |
| 8 | `driver_experience` | Config | Anni esperienza F1 |
| 9 | `session_issues` | Input | Problemi sessione (0/1/2) |
| 10 | `fp2_gap` | FP2 API | Gap FP2 specifico |
| 11 | `sprint_gap` | Sprint API | Gap Sprint Race |

### Architettura Ensemble

```
Feature vector (D1 − D2)
         │
         ├──→  Logistic Regression  ──→  p_lr  × 0.25 ─┐
         ├──→  Random Forest        ──→  p_rf  × 0.38 ─┼──→  Ensemble prob
         └──→  Gradient Boosting    ──→  p_gb  × 0.37 ─┘
```

- Training: ~6.000 gare sintetiche calibrate su F1 2018–2025
- Validazione: Cross-validation 5-fold, metrica AUC-ROC
- AUC tipico: **LR ≈ 0.71 · RF ≈ 0.70 · GBM ≈ 0.70**

### Calcolo Edge

```
Implied prob (corretta per margine) = (1/Q_D1) / (1/Q_D1 + 1/Q_D2)

Edge % = (Prob.Ensemble − Prob.Implicita) × 100

🟢  Edge > +4%   →  Quota SOTTOSTIMATA — valore positivo
🟡  Edge ±4%     →  Quota EQUA — neutrale
🔴  Edge < −5%   →  Quota SOVRASTIMATA — evitare
```

### Mercati supportati

| Mercato | Strategia ML |
|---------|-------------|
| T/T Gara | H2H differenziale D1−D2 |
| T/T 1° Giro | H2H differenziale D1−D2 |
| Migliore Gruppo | H2H round-robin, prob normalizzata nel gruppo |
| Top 6 | Classificatore singolo (RF) |
| Podio | Classificatore singolo (RF) |
| Safety Car | Frequenza storica per circuito |

---

## ✍️ Formato quote testo

```
# T/T Gara  →  "D1 vs D2: quota1 / quota2"
Leclerc vs Norris: 1.25 / 3.50
Russell vs Antonelli: 1.08 / 6.00
Piastri vs Hamilton: 1.83 / 1.83

# T/T Primo Giro
Leclerc vs Norris: 1.30 / 3.20

# Migliore Gruppo  →  "G<n>: Driver=quota, Driver=quota"
G1: Russell=1.25, Antonelli=4.50, Leclerc=7.00, Verstappen=21.00
G2: Piastri=2.50, Hamilton=2.85, Hadjar=3.35, Norris=3.75
G3: Lindblad=2.60, Bearman=3.00, Lawson=3.00, Ocon=4.00
G4: Bortoleto=1.90, Hulkenberg=2.35, Gasly=5.00, Albon=7.00

# Singoli
Hamilton Top6: 1.55
Antonelli Podio: 1.90
Safety Car: 1.60
```

---

## 📷 Quote da screenshot

Carica uno screenshot dell'app Eurobet, Snai, Sisal, ecc.
Claude Vision rileva automaticamente T/T, Gruppi, Top6, Podio, Safety Car.

**Setup:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # da console.anthropic.com (free tier disponibile)
```
Inserisci la chiave anche nella sidebar dell'app.

---

## ⚙️ Configurazione

### Forza costruttori 2026 — `ml/models.py`
```python
DEFAULT_CS = {
    'Russell': 0.97, 'Antonelli': 0.97,   # Mercedes
    'Leclerc': 0.85, 'Hamilton': 0.85,    # Ferrari
    'Piastri': 0.83, 'Norris': 0.83,      # McLaren
    'Hadjar':  0.86, 'Verstappen': 0.86,  # Red Bull
    # ...
}
```

### Storico Safety Car — `core/f1_api.py`
```python
SC_HISTORY = {
    "albert park": [1,1,1,1,0,1,1,1,0,1,1,1],  # 1=SC, 0=no SC, ultimi 12 anni
    "monza":       [1,1,1,0,1,0,1,1,0,1],
    # ...
}
```

### Variabile d'ambiente
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```


## 📦 Dipendenze principali

```
streamlit>=1.32     Web app framework
plotly>=5.18        Grafici interattivi dark-theme
scikit-learn>=1.3   LR + Random Forest + Gradient Boosting
pandas>=2.0         Gestione dati tabellari
numpy>=1.24         Calcoli vettoriali
requests>=2.31      Chiamate OpenF1 + Jolpica API
anthropic>=0.20     Claude Vision (opzionale, per screenshot)
```

---

## ⚠️ Disclaimer

> Questo progetto è sviluppato **esclusivamente a scopo informativo, educativo e di ricerca statistica**.
> Non costituisce in alcun modo un consiglio di scommessa o un invito al gioco d'azzardo.
> Le predizioni ML hanno natura probabilistica e non garantiscono alcun risultato economico.
>
> **Gioca responsabilmente.**
> Numero Verde Nazionale Gioco d'Azzardo: **800 921 121**
> → [giocaresponsabile.it](https://www.giocaresponsabile.it)

---

## 📄 Licenza

MIT — vedi [LICENSE](LICENSE)

---

<div align="center">
  Dati: <a href="https://openf1.org">OpenF1 API</a> + <a href="https://jolpi.ca/ergast">Jolpica/Ergast API</a> · UI: <a href="https://streamlit.io">Streamlit</a> · ML: scikit-learn
</div>

---

## 📈 Valutazione Ex-Post & Ottimizzazione Modello

Una delle funzionalità più avanzate: il **ciclo di feedback** che trasforma ogni gara in dati di training.

### Ciclo di apprendimento

```
   PRIMA della gara          DOPO la gara
   ─────────────────         ──────────────────────────────────
   Inserisci quote      →    Carica risultato via API
   Esegui analisi ML    →    Sistema confronta auto. predizioni
   Registra predizioni  →    Calcola metriche ex-post
                        →    Ottimizza il modello
```

### Metriche ex-post calcolate

| Metrica | Descrizione | Baseline |
|---------|-------------|---------|
| **Accuracy** | % predizioni corrette (soglia 50%) | 50% (random) |
| **Brier Score** | Errore quadratico medio delle probabilità | 0.25 (baseline) |
| **Log Loss** | Penalizza la sovra-confidenza | — |
| **AUC-ROC ex-post** | Area sotto la curva su dati reali | 0.50 (random) |
| **ROI simulato** | Rendimento con stake unitario per fascia edge | 0% (break-even) |
| **Bias di calibrazione** | Media(prob_ML − outcome_reale) | 0% (calibrato) |

### Reliability Diagram (curva di calibrazione)

Il grafico più importante: confronta la probabilità predetta con la frequenza osservata.

```
Prob. predetta    Freq. osservata    Interpretazione
─────────────     ───────────────    ─────────────────────────────────
     80%               80%           ✅ Calibrato — quota fair
     80%               60%           🔴 Ottimista — overconfident
     80%               90%           🟢 Conservativo — edge sottostimato
```


