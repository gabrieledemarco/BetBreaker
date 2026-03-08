#!/usr/bin/env bash
# =============================================================
#  push_to_github.sh — Script per pubblicare su GitHub
#  Uso: bash push_to_github.sh TUO_USERNAME f1-ml-odds-analyzer
# =============================================================
set -e

GITHUB_USER="${1:-TUO_USERNAME}"
REPO_NAME="${2:-f1-ml-odds-analyzer}"
BRANCH="main"

echo ""
echo "🏎  F1 ML Odds Analyzer — GitHub Push"
echo "======================================"
echo "  Utente  : $GITHUB_USER"
echo "  Repo    : $REPO_NAME"
echo "  Branch  : $BRANCH"
echo ""

# ── 1. Verifica git ─────────────────────────────────────────
if ! command -v git &>/dev/null; then
  echo "❌  git non trovato. Installalo prima di continuare."
  exit 1
fi

# ── 2. Inizializza repo locale ──────────────────────────────
git init
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"

# ── 3. Configura identità (se non già configurata) ──────────
if [ -z "$(git config user.email)" ]; then
  read -p "Email GitHub: " GH_EMAIL
  git config user.email "$GH_EMAIL"
fi
if [ -z "$(git config user.name)" ]; then
  read -p "Nome completo: " GH_NAME
  git config user.name "$GH_NAME"
fi

# ── 4. Primo commit ─────────────────────────────────────────
git add .
git commit -m "🏎 Initial commit — F1 ML Odds Analyzer

- Streamlit web app con 5 tab
- OpenF1 API: dati FP1/FP2/FP3 reali
- Jolpica/Ergast API: qualifiche e sprint reali
- Ensemble ML: LR + Random Forest + Gradient Boosting
- Parser quote: testo libero + Claude Vision (screenshot)
- Edge analysis vs probabilità implicita bookmaker
- 5 multiple ML-validated ottimizzate per target vincita
- Grafici Plotly interattivi dark-theme"

# ── 5. Remote e push ────────────────────────────────────────
REMOTE_URL="https://github.com/$GITHUB_USER/$REPO_NAME.git"

if git remote get-url origin &>/dev/null; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

echo ""
echo "📡  Push verso: $REMOTE_URL"
echo "    (ti verrà chiesto username e Personal Access Token)"
echo ""
git push -u origin "$BRANCH"

echo ""
echo "✅  Repository pubblicato!"
echo "    🔗 https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
echo "💡  Prossimi passi:"
echo "    1. Aggiungi screenshot nel README"
echo "    2. Deploya su Streamlit Cloud: https://share.streamlit.io"
echo "       → Connetti il repo → app.py → Deploy!"
