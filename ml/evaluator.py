"""
ml/evaluator.py
Valutazione ex-post e ottimizzazione del modello ML.

Metriche implementate:
  - Accuracy, Precision, Recall
  - Brier Score (calibrazione probabilistica)
  - Log Loss
  - ROI simulato (per soglia edge)
  - Calibration curve (reliability diagram)
  - Edge accuracy (i 🟢 vincono più dei 🔴?)
  - Confusion matrix per fascia edge

Ottimizzazione:
  - Platt Scaling (calibrazione post-hoc isotonica)
  - Aggiornamento dataset sintetico con dati reali
  - Stima constructor_strength da risultati reali
  - Analisi sistematica degli errori (quali feature falliscono)
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import (brier_score_loss, log_loss, roc_auc_score,
                              confusion_matrix, classification_report)

from core.results import PredictionRecord


# ══════════════════════════════════════════════════════════════
# METRICHE EX-POST
# ══════════════════════════════════════════════════════════════

@dataclass
class EvalMetrics:
    n_total:       int
    n_evaluated:   int
    accuracy:      float
    brier_score:   float    # 0=perfetto, 1=pessimo, baseline 0.25
    log_loss_val:  float
    roc_auc:       float
    roi_all:       float    # ROI se si punta tutto (stake=1)
    roi_positive_edge: float  # ROI solo su selezioni con edge > 0
    roi_green:     float    # ROI solo 🟢 (edge > 4%)
    edge_accuracy: Dict[str, float]  # per fascia: {"green": %, "yellow": %, "red": %}
    overconf_bias: float    # media(prob_pred - prob_reale) → se >0 siamo troppo sicuri
    n_by_market:   Dict[str, int]
    acc_by_market: Dict[str, float]


def compute_metrics(records: List[PredictionRecord]) -> Optional[EvalMetrics]:
    """Calcola tutte le metriche ex-post su un set di PredictionRecord valutati."""
    ev = [r for r in records if r.outcome is not None]
    if len(ev) < 3:
        return None

    y_true = np.array([int(r.outcome) for r in ev])
    y_prob  = np.array([r.ensemble_prob for r in ev])
    y_impl  = np.array([r.implied_prob  for r in ev])
    edges   = np.array([r.edge          for r in ev])
    quotas  = np.array([r.quota         for r in ev])

    # ── Metriche base ─────────────────────────────────────────
    y_pred  = (y_prob >= 0.5).astype(int)
    acc     = float(np.mean(y_true == y_pred))
    bs      = float(brier_score_loss(y_true, y_prob))
    ll      = float(log_loss(y_true, np.clip(y_prob, 1e-7, 1-1e-7)))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except:
        auc = 0.5

    # ── ROI simulato (stake unitario su ogni selezione) ────────
    # Profitto = quota-1 se vinto, -1 se perso
    profits = np.where(y_true == 1, quotas - 1.0, -1.0)
    roi_all  = float(np.sum(profits) / len(ev)) * 100

    mask_pos = edges > 0
    roi_pos  = float(np.sum(profits[mask_pos]) / max(mask_pos.sum(), 1)) * 100 if mask_pos.any() else 0.0

    mask_g   = edges > 0.04
    roi_g    = float(np.sum(profits[mask_g]) / max(mask_g.sum(), 1)) * 100 if mask_g.any() else 0.0

    # ── Edge accuracy per fascia ──────────────────────────────
    def _acc_mask(mask):
        if mask.sum() == 0: return float('nan')
        return float(np.mean(y_true[mask]))

    edge_acc = {
        "🟢 Verde (>+4%)":  _acc_mask(edges >  0.04),
        "🟡 Giallo (±4%)": _acc_mask((edges >= -0.04) & (edges <= 0.04)),
        "🔴 Rosso (<-5%)": _acc_mask(edges < -0.05),
    }

    # ── Bias di calibrazione ─────────────────────────────────
    # Se positivo → il modello è troppo ottimista
    overconf = float(np.mean(y_prob - y_true))

    # ── Per mercato ───────────────────────────────────────────
    n_mkt, acc_mkt = {}, {}
    for r in ev:
        mk = r.market
        n_mkt[mk] = n_mkt.get(mk, 0) + 1
    for mk in n_mkt:
        sub = [r for r in ev if r.market == mk]
        acc_mkt[mk] = float(np.mean([int(r.outcome) for r in sub]))

    return EvalMetrics(
        n_total=len(records), n_evaluated=len(ev),
        accuracy=acc, brier_score=bs, log_loss_val=ll, roc_auc=auc,
        roi_all=roi_all, roi_positive_edge=roi_pos, roi_green=roi_g,
        edge_accuracy=edge_acc, overconf_bias=overconf,
        n_by_market=n_mkt, acc_by_market=acc_mkt,
    )


def calibration_data(records: List[PredictionRecord],
                     n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ritorna (prob_pred, prob_true, counts) per il reliability diagram.
    Confronta anche vs prob. implicita bookmaker.
    """
    ev = [r for r in records if r.outcome is not None]
    if len(ev) < 5:
        return np.array([]), np.array([]), np.array([])

    y_true = np.array([int(r.outcome) for r in ev])
    y_prob  = np.array([r.ensemble_prob for r in ev])

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    counts = np.histogram(y_prob, bins=n_bins, range=(0,1))[0]
    return prob_pred, prob_true, counts


def roi_by_edge_threshold(records: List[PredictionRecord],
                           thresholds: np.ndarray = None) -> pd.DataFrame:
    """
    Simula il ROI per ogni soglia edge (punta solo se edge > soglia).
    Utile per trovare la soglia ottimale.
    """
    ev = [r for r in records if r.outcome is not None]
    if not ev:
        return pd.DataFrame()
    if thresholds is None:
        thresholds = np.arange(-0.10, 0.30, 0.01)

    y_true  = np.array([int(r.outcome) for r in ev])
    profits = np.array([r.quota - 1.0 if r.outcome else -1.0 for r in ev])
    edges   = np.array([r.edge for r in ev])

    rows = []
    for thr in thresholds:
        mask = edges >= thr
        n = mask.sum()
        if n == 0:
            continue
        roi     = float(np.sum(profits[mask]) / n) * 100
        win_rate = float(np.mean(y_true[mask])) * 100
        rows.append({"edge_threshold": round(thr*100, 1), "n_bets": int(n),
                     "roi_pct": round(roi, 2), "win_rate_pct": round(win_rate, 1)})
    return pd.DataFrame(rows)


def edge_vs_accuracy_bins(records: List[PredictionRecord],
                           n_bins: int = 8) -> pd.DataFrame:
    """
    Raggruppa le predizioni per fascia edge e calcola win-rate reale.
    Mostra se edge alto → win rate effettivamente più alto.
    """
    ev = [r for r in records if r.outcome is not None]
    if len(ev) < n_bins:
        return pd.DataFrame()

    edges   = np.array([r.edge for r in ev])
    y_true  = np.array([int(r.outcome) for r in ev])
    y_prob  = np.array([r.ensemble_prob for r in ev])
    y_impl  = np.array([r.implied_prob for r in ev])

    bins  = np.percentile(edges, np.linspace(0, 100, n_bins+1))
    rows  = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask   = (edges >= lo) & (edges <= hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "Edge range":   f"{lo*100:+.1f}% → {hi*100:+.1f}%",
            "N":            int(mask.sum()),
            "Win rate %":   round(float(np.mean(y_true[mask]))*100, 1),
            "Prob ML avg %":round(float(np.mean(y_prob[mask]))*100, 1),
            "Prob impl avg %": round(float(np.mean(y_impl[mask]))*100, 1),
            "ROI %":        round(float(np.mean(
                np.where(y_true[mask]==1,
                         np.array([r.quota for r in ev])[mask]-1,
                         -1.0)
            ))*100, 1),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# OTTIMIZZAZIONE MODELLO
# ══════════════════════════════════════════════════════════════

class ModelOptimizer:
    """
    Ottimizza il modello usando le predizioni valutate con risultati reali.

    Strategie:
    1. Platt Scaling / Isotonic Regression per calibrazione post-hoc
    2. Stima empirica della forza costruttori dai risultati reali
    3. Feature importance analysis: quali feature predicono meglio
    4. Errore sistematico per tipo di mercato
    5. Dataset augmentation: aggiunge dati reali al training sintetico
    """

    def __init__(self, records: List[PredictionRecord] = None, real_only: bool = True):
        """
        real_only=True (default): usa solo record con source='real'.
        Imposta real_only=False SOLO per debug o test controllati.
        """
        all_r = records or []
        if real_only:
            self.records = [r for r in all_r if getattr(r, 'source', 'real') == 'real']
        else:
            self.records = all_r
        self.real_only = real_only
        self.calibrator = None
        self._calibrated = False

    def add_records(self, new_records: List[PredictionRecord]):
        existing_labels = {(r.year, r.round_num, r.label) for r in self.records}
        to_add = new_records
        if self.real_only:
            to_add = [r for r in new_records if getattr(r, 'source', 'real') == 'real']
        self.records += [r for r in to_add
                         if (r.year, r.round_num, r.label) not in existing_labels]

    # ── 1. Calibrazione isotonica ─────────────────────────────

    def fit_calibration(self) -> bool:
        """
        Isotonic regression: mappa prob ML → prob calibrata.
        Richiede almeno 20 osservazioni per essere affidabile.
        """
        ev = [r for r in self.records if r.outcome is not None]
        if len(ev) < 20:
            return False
        y_true = np.array([int(r.outcome) for r in ev])
        y_prob  = np.array([r.ensemble_prob for r in ev])
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(y_prob, y_true)
        self._calibrated = True
        return True

    def calibrated_prob(self, raw_prob: float) -> float:
        if not self._calibrated or self.calibrator is None:
            return raw_prob
        return float(self.calibrator.predict([raw_prob])[0])

    def calibration_improvement(self) -> Dict[str, float]:
        """Confronta Brier Score prima/dopo calibrazione."""
        ev = [r for r in self.records if r.outcome is not None]
        if len(ev) < 5:
            return {}
        y_true = np.array([int(r.outcome) for r in ev])
        y_raw  = np.array([r.ensemble_prob for r in ev])
        bs_raw = brier_score_loss(y_true, y_raw)
        if self._calibrated:
            y_cal  = np.array([self.calibrated_prob(p) for p in y_raw])
            bs_cal = brier_score_loss(y_true, y_cal)
        else:
            bs_cal = bs_raw
        return {"brier_raw": bs_raw, "brier_calibrated": bs_cal,
                "improvement": bs_raw - bs_cal}

    # ── 2. Stima constructor_strength dai risultati ───────────

    def estimate_constructor_strength(self) -> Dict[str, float]:
        """
        Calcola la forza costruttore osservata = win_rate H2H pesato per
        la differenza di posizione in griglia.
        """
        ev = [r for r in self.records
              if r.outcome is not None and r.market == "T/T Gara" and r.driver2]
        if len(ev) < 10:
            return {}

        # Accoppiamento teammate: D1 e D2 nella stessa scuderia
        # Non sappiamo la scuderia qui — usiamo approccio empirico diretto
        # per pilota: % di H2H vinti
        driver_wins  = {}
        driver_total = {}
        for r in ev:
            d1, d2 = r.driver1, r.driver2
            for d in [d1, d2]:
                driver_wins[d]  = driver_wins.get(d, 0)
                driver_total[d] = driver_total.get(d, 0)
            if r.outcome:  # D1 ha vinto
                driver_wins[d1] += 1
            else:
                driver_wins[d2] += 1
            driver_total[d1] += 1
            driver_total[d2] += 1

        winrates = {}
        for d in driver_total:
            if driver_total[d] >= 3:
                wr = driver_wins[d] / driver_total[d]
                # Normalizza su scala 0.3–1.0
                winrates[d] = round(0.30 + wr * 0.70, 3)
        return dict(sorted(winrates.items(), key=lambda x: -x[1]))

    # ── 3. Analisi errori sistematici ────────────────────────

    def systematic_errors(self) -> pd.DataFrame:
        """
        Per ogni feature identifica se il modello sbaglia sistematicamente.
        Ritorna tabella con mean_error per fascia di prob predetta.
        """
        ev = [r for r in self.records if r.outcome is not None]
        if len(ev) < 10:
            return pd.DataFrame()

        rows = []
        # Raggruppa per: mercato, fascia di quota
        for mk in set(r.market for r in ev):
            sub = [r for r in ev if r.market == mk]
            if len(sub) < 3:
                continue
            probs = np.array([r.ensemble_prob for r in sub])
            trues = np.array([int(r.outcome)  for r in sub])
            edges = np.array([r.edge          for r in sub])
            quotas= np.array([r.quota         for r in sub])

            # Bias = modello prevede troppo alto o troppo basso?
            bias  = float(np.mean(probs - trues))
            # Accuracy
            acc   = float(np.mean((probs >= 0.5) == trues.astype(bool)))
            # Brier
            bs    = float(brier_score_loss(trues, probs))
            # ROI
            profits = np.where(trues==1, quotas-1, -1.0)
            roi   = float(np.mean(profits)) * 100

            rows.append({
                "Mercato": mk,
                "N": len(sub),
                "Accuracy %": round(acc*100, 1),
                "Brier Score": round(bs, 4),
                "Bias (prob-reale)": round(bias, 4),
                "ROI %": round(roi, 2),
                "Edge medio %": round(float(np.mean(edges))*100, 2),
            })
        return pd.DataFrame(rows).sort_values("ROI %", ascending=False)

    # ── 4. Dataset reale per retraining ──────────────────────

    def build_real_training_rows(self) -> pd.DataFrame:
        """
        Converte le predizioni valutate in righe per il training ML.
        Formato: feature differenziali + esito reale.
        """
        ev = [r for r in self.records
              if r.outcome is not None and r.market == "T/T Gara"
              and r.driver2 and r.outcome_position_d1 and r.outcome_position_d2]
        if not ev:
            return pd.DataFrame()

        rows = []
        for r in ev:
            # Feature proxy: usiamo le probabilità già calcolate come feature ridotte
            # In un sistema completo useremmo le feature originali
            rows.append({
                "prob_ml": r.ensemble_prob,
                "implied_prob": r.implied_prob,
                "edge": r.edge,
                "quota": r.quota,
                "year": r.year,
                "round": r.round_num,
                "market": r.market,
                "win": int(r.outcome),
                # posizioni come proxy feature
                "pos_d1": r.outcome_position_d1 or 99,
                "pos_d2": r.outcome_position_d2 or 99,
            })
        return pd.DataFrame(rows)

    # ── 5. Riepilogo raccomandazioni ─────────────────────────

    def optimization_recommendations(self) -> List[Dict]:
        """Genera raccomandazioni concrete per migliorare il modello."""
        ev = [r for r in self.records if r.outcome is not None]
        recs = []
        n = len(ev)

        if n < 10:
            recs.append({
                "priorità": "⏳ DATI INSUFFICIENTI",
                "raccomandazione": f"Accumula almeno 30 predizioni valutate (hai {n}). "
                                   "Ogni GP aggiunge ~15-30 osservazioni.",
                "azione": "Continua ad usare l'app per ogni GP e registra i risultati."
            })
            return recs

        metrics = compute_metrics(ev)
        if not metrics:
            return recs

        # Bias di calibrazione
        if abs(metrics.overconf_bias) > 0.05:
            direction = "ottimista" if metrics.overconf_bias > 0 else "pessimista"
            recs.append({
                "priorità": "🔴 ALTA — Calibrazione",
                "raccomandazione": f"Il modello è sistematicamente {direction} "
                                   f"(bias medio: {metrics.overconf_bias*100:+.1f}%). "
                                   "Applica la calibrazione isotonica.",
                "azione": f"optimizer.fit_calibration()  →  brier improvement atteso: "
                          f"{abs(metrics.overconf_bias)*0.05:.4f}"
            })

        # ROI per fascia
        roi_g = metrics.roi_green
        if roi_g < 0:
            recs.append({
                "priorità": "🔴 ALTA — Edge Threshold",
                "raccomandazione": f"Le selezioni 🟢 hanno ROI negativo ({roi_g:.1f}%). "
                                   "La soglia +4% è troppo bassa per questo bookmaker.",
                "azione": "Aumenta la soglia minima a +8% o +10%. "
                          "Usa il grafico ROI-by-threshold per trovare il punto ottimale."
            })
        elif roi_g > 10:
            recs.append({
                "priorità": "🟢 POSITIVO — Edge Soglia",
                "raccomandazione": f"Le selezioni 🟢 hanno ROI positivo ({roi_g:.1f}%). "
                                   "Il modello sta identificando valore reale.",
                "azione": "Considera di abbassare la soglia a +2% per aumentare la copertura."
            })

        # Mercati con ROI negativo
        se = self.systematic_errors()
        if not se.empty:
            worst = se.sort_values("ROI %").iloc[0]
            recs.append({
                "priorità": "🟡 MEDIA — Mercato peggiore",
                "raccomandazione": f"Il mercato '{worst['Mercato']}' ha ROI {worst['ROI %']:.1f}% "
                                   f"e accuracy {worst['Accuracy %']:.1f}%.",
                "azione": f"Considera di escludere o de-pesare questo mercato nell'ensemble."
            })

        # Feature: aggiorna constructor strength
        cs_est = self.estimate_constructor_strength()
        if cs_est:
            recs.append({
                "priorità": "🟡 MEDIA — Constructor Strength",
                "raccomandazione": "Dati reali sufficienti per stimare la forza costruttori empirica.",
                "azione": f"Aggiorna ml/models.py DEFAULT_CS con: {dict(list(cs_est.items())[:5])}"
            })

        # Calibrazione disponibile
        if n >= 20 and not self._calibrated:
            recs.append({
                "priorità": "🟡 MEDIA — Calibrazione disponibile",
                "raccomandazione": f"Con {n} osservazioni puoi applicare la calibrazione isotonica.",
                "azione": "Premi il bottone 'Calibra modello' nel tab Ottimizzazione."
            })

        return recs