"""
core/results.py
Recupera i risultati reali di gara, qualifiche e sprint da Jolpica/Ergast API.
Fornisce anche un layer di persistenza locale (JSON) per accumulare storico.
"""
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
TIMEOUT = 12
CACHE_DIR = Path("data/results_cache")


# ══════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════

@dataclass
class DriverResult:
    driver:      str
    position:    int          # posizione finale (99 = DNF/DNS)
    grid:        int          # posizione di partenza
    points:      float
    status:      str          # "Finished", "Retired", "+1 Lap", ecc.
    fastest_lap: bool = False

    @property
    def finished(self) -> bool:
        return self.position <= 20 and "Retired" not in self.status

    @property
    def podium(self) -> bool:
        return self.position <= 3

    @property
    def top6(self) -> bool:
        return self.position <= 6


@dataclass
class RaceResult:
    year:        int
    round_num:   int
    event_name:  str
    circuit:     str
    date:        str
    results:     Dict[str, DriverResult] = field(default_factory=dict)
    h2h_outcomes: Dict[str, bool]        = field(default_factory=dict)
    # h2h_outcomes: "D1 vs D2" → True se D1 ha battuto D2


@dataclass
class PredictionRecord:
    """Un'unica predizione registrata con il risultato reale."""
    year:         int
    round_num:    int
    event_name:   str
    market:       str         # "T/T Gara", "Migliore Gruppo", ecc.
    label:        str         # "Leclerc batte Norris"
    driver1:      str
    driver2:      Optional[str]
    quota:        float
    implied_prob: float
    ensemble_prob: float
    edge:         float
    # Risultato reale
    outcome:      Optional[bool]  = None   # True=vinto, False=perso, None=sconosciuto
    outcome_position_d1: Optional[int] = None
    outcome_position_d2: Optional[int] = None
    # Sorgente: "real" = dati veri, "demo" = test/simulazione
    source:       str              = "real"


# ══════════════════════════════════════════════════════════════
# FETCH RISULTATI GARA
# ══════════════════════════════════════════════════════════════

def _get(url: str, params: dict = None) -> Optional[dict]:
    for attempt in range(2):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 0:
                time.sleep(1.5)
    return None


def fetch_race_result(year: int, round_num: int) -> Optional[RaceResult]:
    """Recupera il risultato di gara completo da Jolpica."""
    data = _get(f"{JOLPICA_BASE}/{year}/{round_num}/results.json")
    if not data:
        return None

    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return None

    race_raw = races[0]
    rr = RaceResult(
        year=year,
        round_num=round_num,
        event_name=race_raw.get("raceName", ""),
        circuit=race_raw.get("Circuit", {}).get("circuitName", ""),
        date=race_raw.get("date", ""),
    )

    for r in race_raw.get("Results", []):
        drv  = _norm(r["Driver"]["familyName"])
        pos  = int(r.get("position", 99))
        grid = int(r.get("grid", 99))
        pts  = float(r.get("points", 0))
        status = r.get("status", "Unknown")
        fl = r.get("FastestLap", {}).get("rank", "99") == "1"
        rr.results[drv] = DriverResult(drv, pos, grid, pts, status, fl)

    # Costruisci H2H outcomes per tutti i driver pair
    for d1, r1 in rr.results.items():
        for d2, r2 in rr.results.items():
            if d1 < d2:  # evita duplicati
                key = f"{d1} vs {d2}"
                rr.h2h_outcomes[key] = r1.position < r2.position

    return rr


def fetch_sprint_result(year: int, round_num: int) -> Optional[RaceResult]:
    """Recupera il risultato Sprint da Jolpica."""
    data = _get(f"{JOLPICA_BASE}/{year}/{round_num}/sprint.json")
    if not data:
        return None
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return None

    race_raw = races[0]
    rr = RaceResult(
        year=year, round_num=round_num,
        event_name=race_raw.get("raceName","") + " Sprint",
        circuit=race_raw.get("Circuit",{}).get("circuitName",""),
        date=race_raw.get("date",""),
    )
    for r in race_raw.get("SprintResults", []):
        drv  = _norm(r["Driver"]["familyName"])
        pos  = int(r.get("position", 99))
        grid = int(r.get("grid", 99))
        pts  = float(r.get("points", 0))
        rr.results[drv] = DriverResult(drv, pos, grid, pts, r.get("status",""), False)
    return rr


def _norm(name: str) -> str:
    MAP = {
        "russell":"Russell","antonelli":"Antonelli","leclerc":"Leclerc",
        "hamilton":"Hamilton","piastri":"Piastri","norris":"Norris",
        "hadjar":"Hadjar","verstappen":"Verstappen","lawson":"Lawson",
        "lindblad":"Lindblad","bortoleto":"Bortoleto","hulkenberg":"Hulkenberg",
        "ocon":"Ocon","bearman":"Bearman","gasly":"Gasly","colapinto":"Colapinto",
        "albon":"Albon","sainz":"Sainz","bottas":"Bottas","perez":"Perez",
        "alonso":"Alonso","stroll":"Stroll",
    }
    return MAP.get(name.lower(), name.capitalize())


# ══════════════════════════════════════════════════════════════
# MATCH PREDIZIONI → RISULTATI
# ══════════════════════════════════════════════════════════════

def evaluate_predictions(preds_records: List[PredictionRecord],
                          race_result: RaceResult) -> List[PredictionRecord]:
    """
    Per ogni predizione, cerca il risultato reale e aggiorna outcome.
    Ritorna la lista aggiornata.
    """
    updated = []
    for p in preds_records:
        p2 = PredictionRecord(**asdict(p))  # copia

        d1 = p.driver1
        r1 = race_result.results.get(d1)

        if p.market == "T/T Gara" and p.driver2:
            d2 = p.driver2
            r2 = race_result.results.get(d2)
            if r1 and r2:
                p2.outcome = r1.position < r2.position
                p2.outcome_position_d1 = r1.position
                p2.outcome_position_d2 = r2.position

        elif p.market == "Migliore Gruppo":
            if r1:
                p2.outcome_position_d1 = r1.position

        elif p.market in ("Top 6", "Top6"):
            if r1:
                p2.outcome = r1.top6
                p2.outcome_position_d1 = r1.position

        elif p.market == "Podio":
            if r1:
                p2.outcome = r1.podium
                p2.outcome_position_d1 = r1.position

        updated.append(p2)
    return updated


# ══════════════════════════════════════════════════════════════
# PERSISTENZA STORICO LOCALE
# ══════════════════════════════════════════════════════════════

def _history_path(year: int = None) -> str:
    """Percorso file storico per anno. Ogni anno ha il suo file separato."""
    if year:
        return f"data/history_{year}.json"
    return "data/history.json"


def save_records(records: List[PredictionRecord], path: str = None, year: int = None):
    if path is None:
        path = _history_path(year or (records[0].year if records else None))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    existing = load_records(path)
    # Deduplicazione per (year, round, label)
    seen = {(r.year, r.round_num, r.label) for r in existing}
    new = [r for r in records if (r.year, r.round_num, r.label) not in seen]
    all_records = existing + new
    with open(path, "w") as f:
        json.dump([asdict(r) for r in all_records], f, indent=2)
    return len(new)


def delete_records(source: str = None, year: int = None,
                   round_num: int = None, path: str = None) -> int:
    if path is None:
        path = _history_path(year)
    """
    Elimina record dallo storico con filtri combinabili.
    Es: delete_records(source='demo') → elimina tutti i record demo
        delete_records(year=2026, round_num=1) → elimina GP Australia 2026
    Ritorna il numero di record eliminati.
    """
    existing = load_records(path)
    before = len(existing)
    kept = []
    for r in existing:
        remove = True
        if source   is not None and r.source   != source:     remove = False
        if year     is not None and r.year      != year:       remove = False
        if round_num is not None and r.round_num != round_num: remove = False
        if not remove:
            kept.append(r)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in kept], f, indent=2)
    return before - len(kept)


def get_real_records(path: str = None, year: int = None) -> List[PredictionRecord]:
    if path is None:
        path = _history_path(year)
    """Solo i record con source='real' — esclude test e demo."""
    return [r for r in load_records(path) if getattr(r, 'source', 'real') == 'real']


def list_history_files() -> List[str]:
    """Elenca tutti i file storico disponibili."""
    import glob
    files = glob.glob("data/history_*.json") + (["data/history.json"] if Path("data/history.json").exists() else [])
    return sorted(files)


def history_summary(path: str = None, year: int = None) -> dict:
    """Riepilogo dello storico per sorgente. Se year=None, aggrega tutti gli anni."""
    if path is None and year is None:
        # Aggrega tutti i file storico disponibili
        all_r = []
        for f in list_history_files():
            all_r.extend(load_records(f))
    else:
        all_r = load_records(path, year)
    real  = [r for r in all_r if getattr(r, 'source', 'real') == 'real']
    demo  = [r for r in all_r if getattr(r, 'source', 'real') == 'demo']
    ev_real = [r for r in real if r.outcome is not None]
    ev_demo = [r for r in demo if r.outcome is not None]
    return {
        "total": len(all_r),
        "real": len(real), "real_evaluated": len(ev_real),
        "demo": len(demo), "demo_evaluated": len(ev_demo),
    }


def load_records(path: str = None, year: int = None) -> List[PredictionRecord]:
    if path is None:
        path = _history_path(year)
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        raw = json.load(f)
    records = []
    for r in raw:
        # Retrocompatibilità: record senza campo 'source' → assume 'real'
        r.setdefault('source', 'real')
        try:
            records.append(PredictionRecord(**r))
        except TypeError:
            pass  # ignora record con schema diverso
    return records


def get_evaluated_records(path: str = "data/history.json") -> List[PredictionRecord]:
    """Solo le predizioni con outcome noto."""
    return [r for r in load_records(path) if r.outcome is not None]