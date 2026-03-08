"""
ml/models.py — Training LR + RF + GBM + predizioni + edge
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Optional, Tuple
import warnings; warnings.filterwarnings('ignore')

FEATURE_NAMES = [
    'grid_pos', 'quali_gap', 'avg_fp_gap', 'best_fp_gap',
    'fp_consistency', 'trend_improvement', 'constructor_strength',
    'driver_experience', 'session_issues', 'fp2_gap', 'sprint_gap'
]

# Forza costruttori 2026 (aggiornabile)
DEFAULT_CS = {
    'Russell': 0.97, 'Antonelli': 0.97,
    'Leclerc': 0.85, 'Hamilton': 0.85,
    'Piastri': 0.83, 'Norris': 0.83,
    'Hadjar': 0.86, 'Verstappen': 0.86,
    'Lawson': 0.62, 'Lindblad': 0.62,
    'Bortoleto': 0.59, 'Hulkenberg': 0.59,
    'Ocon': 0.57, 'Bearman': 0.57,
    'Gasly': 0.54, 'Colapinto': 0.54,
    'Albon': 0.52, 'Sainz': 0.52,
    'Bottas': 0.39, 'Perez': 0.39,
    'Alonso': 0.34, 'Stroll': 0.34,
}

DEFAULT_EXP = {
    'Russell': 7, 'Antonelli': 0.5, 'Leclerc': 7, 'Hamilton': 18,
    'Piastri': 3, 'Norris': 6, 'Hadjar': 0.5, 'Verstappen': 10,
    'Lawson': 2, 'Lindblad': 0.2, 'Bortoleto': 0.5, 'Hulkenberg': 12,
    'Ocon': 7, 'Bearman': 1, 'Gasly': 8, 'Colapinto': 0.5,
    'Albon': 6, 'Sainz': 10, 'Bottas': 14, 'Perez': 14,
    'Alonso': 22, 'Stroll': 7,
}


def _build_h2h_dataset(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        g1, g2 = rng.integers(1,23), rng.integers(1,23)
        q1 = min(rng.exponential(0.4 if g1<=8 else 1.2), 5.0)
        q2 = min(rng.exponential(0.4 if g2<=8 else 1.2), 5.0)
        a1,a2 = q1*rng.uniform(0.9,1.4), q2*rng.uniform(0.9,1.4)
        b1,b2 = a1*rng.uniform(0.7,1.0), a2*rng.uniform(0.7,1.0)
        c1,c2 = rng.exponential(0.3), rng.exponential(0.3)
        t1,t2 = rng.uniform(-1,2), rng.uniform(-1,2)
        cs1,cs2 = rng.uniform(0.3,1.0), rng.uniform(0.3,1.0)
        e1,e2   = rng.uniform(0,22), rng.uniform(0,22)
        i1,i2   = rng.integers(0,3), rng.integers(0,3)
        fp2_1,fp2_2 = q1*rng.uniform(0.9,1.3), q2*rng.uniform(0.9,1.3)
        spr1,spr2   = rng.uniform(0,2.5), rng.uniform(0,2.5)
        s1 = -g1*0.08 - q1*0.15 + cs1*0.4 + e1*0.01 - i1*0.12 - a1*0.05 + t1*0.05 - spr1*0.03
        s2 = -g2*0.08 - q2*0.15 + cs2*0.4 + e2*0.01 - i2*0.12 - a2*0.05 + t2*0.05 - spr2*0.03
        p = 1/(1+np.exp(-(s1-s2)+rng.normal(0,0.3)))
        rows.append([g1-g2,q1-q2,a1-a2,b1-b2,c1-c2,t1-t2,cs1-cs2,e1-e2,i1-i2,fp2_1-fp2_2,spr1-spr2,int(rng.random()<p)])
    return pd.DataFrame(rows, columns=FEATURE_NAMES+['win'])


def _build_single_dataset(target='top6', n=6000, seed=77):
    rng = np.random.default_rng(seed)
    base = {'podium':{1:.54,2:.33,3:.22,4:.14,5:.10,6:.08},
            'top6':  {1:.82,2:.68,3:.58,4:.48,5:.42,6:.36}}
    br = base.get(target, base['top6'])
    rows = []
    for _ in range(n):
        g  = rng.integers(1,23)
        q  = min(rng.exponential(0.4 if g<=8 else 1.2), 5.0)
        a  = q*rng.uniform(0.9,1.4); b=a*rng.uniform(0.7,1.0)
        c  = rng.exponential(0.3); t=rng.uniform(-1,2)
        cs = rng.uniform(0.3,1.0); exp=rng.uniform(0,22); iss=rng.integers(0,3)
        fp2= q*rng.uniform(0.9,1.3); spr=rng.uniform(0,2.5)
        bp = br.get(g, max(0.001, 0.54-g*0.025))
        p  = min(bp*(0.5+cs*0.8)*(0.8+min(exp/20,0.3))*max(0.1,1-iss*0.2)*rng.uniform(0.7,1.3),0.95)
        rows.append([g,q,a,b,c,t,cs,exp,iss,fp2,spr,int(rng.random()<p)])
    return pd.DataFrame(rows, columns=FEATURE_NAMES+[target])


class F1MLEngine:
    """Motore ML completo: training + predizioni + edge."""

    def __init__(self):
        self._trained = False
        self.cv: Dict[str,float] = {}
        self.scaler_h2h = StandardScaler()
        self.scaler_s   = StandardScaler()

    def train(self) -> Dict[str, float]:
        # H2H
        df  = _build_h2h_dataset()
        X   = df[FEATURE_NAMES].values; y = df['win'].values
        Xs  = self.scaler_h2h.fit_transform(X)
        self.lr = LogisticRegression(C=0.5, max_iter=2000).fit(Xs, y)
        self.rf = RandomForestClassifier(400, max_depth=9, min_samples_leaf=8, random_state=42).fit(X, y)
        self.gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.04, max_depth=4, subsample=0.8, random_state=42).fit(X, y)
        self.cv['h2h_lr'] = cross_val_score(self.lr,Xs,y,cv=5,scoring='roc_auc').mean()
        self.cv['h2h_rf'] = cross_val_score(self.rf,X,y,cv=5,scoring='roc_auc').mean()
        self.cv['h2h_gb'] = cross_val_score(self.gb,X,y,cv=5,scoring='roc_auc').mean()
        # Top6
        df6  = _build_single_dataset('top6')
        X6   = df6[FEATURE_NAMES].values; y6 = df6['top6'].values
        Xs6  = self.scaler_s.fit_transform(X6)
        self.rf6 = RandomForestClassifier(400, max_depth=8, min_samples_leaf=8, random_state=42).fit(X6, y6)
        self.cv['top6_rf'] = cross_val_score(self.rf6,X6,y6,cv=5,scoring='roc_auc').mean()
        self._trained = True
        return self.cv

    def _ens_h2h(self, xd: np.ndarray) -> Tuple[float,float,float,float]:
        xs  = self.scaler_h2h.transform(xd)
        lr  = self.lr.predict_proba(xs)[0][1]
        rf  = self.rf.predict_proba(xd)[0][1]
        gb  = self.gb.predict_proba(xd)[0][1]
        return lr, rf, gb, lr*0.25+rf*0.38+gb*0.37

    def predict_h2h(self, f1: np.ndarray, f2: np.ndarray) -> Tuple[float,float,float,float]:
        xd = (f1-f2).reshape(1,-1)
        return self._ens_h2h(xd)

    def predict_top6(self, f: np.ndarray) -> float:
        x  = f.reshape(1,-1)
        xs = self.scaler_s.transform(x)
        return self.rf6.predict_proba(x)[0][1]

    def feature_importance(self) -> pd.Series:
        labels = ['Griglia','Quali Gap','Avg FP','Best FP','Consistenza',
                  'Trend FP','Costruttore','Esperienza','Incidenti','FP2 Gap','Sprint Gap']
        return pd.Series(self.rf.feature_importances_, index=labels).sort_values(ascending=False)


def build_driver_features(driver: str,
                           sessions: Dict[str, Dict[str,float]],
                           grid: Dict[str,int],
                           issues: Dict[str,int] = None,
                           cs: Dict[str,float] = None,
                           exp: Dict[str,float] = None) -> np.ndarray:
    """Costruisce il vettore feature per un singolo pilota."""
    cs  = cs  or DEFAULT_CS
    exp = exp or DEFAULT_EXP
    iss = issues or {}

    fp_vals = [sessions.get(s,{}).get(driver, 5.0) for s in ['FP1','FP2','FP3'] if s in sessions]
    fp_vals = [v for v in fp_vals if v < 4.9]
    fp2_gap = sessions.get('FP2',{}).get(driver, 5.0)
    fp1_gap = sessions.get('FP1',{}).get(driver, 5.0)
    fp3_gap = sessions.get('FP3',{}).get(driver, 5.0)
    spr_gap = sessions.get('Sprint',{}).get(driver, 5.0)
    q_gap   = sessions.get('Quali',{}).get(driver, 5.0)

    avg_fp   = float(np.mean(fp_vals)) if fp_vals else 4.0
    best_fp  = float(min(fp_vals)) if fp_vals else 4.0
    consist  = float(np.std(fp_vals)) if len(fp_vals)>1 else 2.0
    trend    = fp1_gap - fp3_gap if fp1_gap<4.9 and fp3_gap<4.9 else 0.0

    return np.array([
        float(grid.get(driver, 15)),
        min(q_gap, 5.0),
        avg_fp, best_fp, consist, trend,
        cs.get(driver, 0.5),
        exp.get(driver, 5.0),
        float(iss.get(driver, 0)),
        min(fp2_gap, 5.0),
        min(spr_gap, 5.0),
    ])
