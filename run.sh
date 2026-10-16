#!/usr/bin/env bash
# Lance le projet de zéro : dépendances, vérifications, base de données,
# puis MLflow (5000), Streamlit (8501) et le chat Gradio (7860).
#
#   ./run.sh               installe, vérifie et lance tout
#   ./run.sh --check       vérifications seulement, sans rien lancer
#   ./run.sh --rebuild-db  supprime et recrée data/database.db depuis Kaggle
#   ./run.sh --no-chat     ne lance pas le chat Gradio
#
# Ports modifiables : MLFLOW_PORT=5001 STREAMLIT_PORT=8502 GRADIO_PORT=7861 ./run.sh
set -euo pipefail

cd "$(dirname "$0")"

MLFLOW_PORT="${MLFLOW_PORT:-5000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
GRADIO_PORT="${GRADIO_PORT:-7860}"
DB_PATH="data/database.db"
LOG_DIR="logs"
TABLES=(dim_rencontre dim_powerlifter dim_division dim_equipement participation_rencontre fait_performance)

CHECK_ONLY=false
REBUILD_DB=false
WITH_CHAT=true
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=true ;;
        --rebuild-db) REBUILD_DB=true ;;
        --no-chat) WITH_CHAT=false ;;
        -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Option inconnue : $arg (voir --help)" >&2; exit 2 ;;
    esac
done

if [ -t 1 ]; then
    BOLD=$'\e[1m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; RED=$'\e[31m'; RESET=$'\e[0m'
else
    BOLD=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi
step() { echo; echo "${BOLD}==> $*${RESET}"; }
ok()   { echo "  ${GREEN}✓${RESET} $*"; }
warn() { echo "  ${YELLOW}!${RESET} $*"; WARNINGS=$((WARNINGS + 1)); }
fail() { echo "  ${RED}✗${RESET} $*"; ERRORS=$((ERRORS + 1)); }
die()  { echo "${RED}Erreur :${RESET} $*" >&2; exit 1; }
WARNINGS=0
ERRORS=0

# Port occupé ? (bash pur, sans ss/lsof)
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
mlflow_up() { curl -sf -m 2 "http://127.0.0.1:$MLFLOW_PORT/health" >/dev/null; }

# --- 1. Prérequis --------------------------------------------------------------
step "Prérequis"
command -v uv >/dev/null || die "uv est introuvable. Installe-le : curl -LsSf https://astral.sh/uv/install.sh | sh"
ok "uv $(uv --version | cut -d' ' -f2)"
command -v curl >/dev/null || die "curl est introuvable."
ok "curl"

for f in pyproject.toml app.py common.py etl.py ml.py groq_chat.py data/sql/create_tables.sql; do
    [ -f "$f" ] || fail "Fichier manquant : $f"
done
[ -d pages ] || fail "Dossier manquant : pages/"
[ "$ERRORS" -eq 0 ] || die "Fichiers du projet manquants, arrêt."
ok "Fichiers du projet présents"

# --- 2. Dépendances ------------------------------------------------------------
step "Dépendances (uv sync)"
uv sync --quiet || die "Échec de l'installation des dépendances."
ok "Environnement .venv à jour ($(uv run --no-sync python --version))"

PY=(uv run --no-sync python)
"${PY[@]}" -c "import streamlit, mlflow, sklearn, pandas, plotly, gradio, groq, kagglehub, dotenv" 2>/dev/null \
    || die "Des modules Python manquent malgré uv sync."
ok "Modules Python importables"

# --- 3. Configuration ----------------------------------------------------------
step "Configuration (.env)"
GROQ_API_KEY="${GROQ_API_KEY:-}"
if [ -z "$GROQ_API_KEY" ] && [ -f .env ]; then
    GROQ_API_KEY="$(grep -E '^GROQ_API_KEY=' .env | tail -1 | cut -d= -f2- | tr -d '"'"'"' ')"
fi
if ! $WITH_CHAT; then
    ok "Chat désactivé (--no-chat)"
elif [ -z "$GROQ_API_KEY" ]; then
    warn "GROQ_API_KEY absente (.env ou environnement) : le chat Gradio ne sera pas lancé."
    WITH_CHAT=false
else
    code="$(curl -s -o /dev/null -w '%{http_code}' -m 5 -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/openai/v1/models || true)"
    case "$code" in
        200) ok "GROQ_API_KEY valide" ;;
        401|403) warn "GROQ_API_KEY refusée par Groq (HTTP $code) : le chat ne sera pas lancé."; WITH_CHAT=false ;;
        *) warn "Groq injoignable (HTTP ${code:-000}) : le chat sera lancé mais pourra échouer." ;;
    esac
fi

# --- 4. Base de données --------------------------------------------------------
step "Base de données ($DB_PATH)"
db_ok() {
    "${PY[@]}" - "$DB_PATH" "${TABLES[@]}" <<'EOF'
import sqlite3, sys
db, tables = sys.argv[1], sys.argv[2:]
try:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    vides = [t for t in tables if con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0]
except sqlite3.Error as e:
    sys.exit(f"base illisible ({e})")
if vides:
    sys.exit(f"tables vides : {', '.join(vides)}")
print(con.execute("SELECT COUNT(*) FROM fait_performance").fetchone()[0])
EOF
}

if $REBUILD_DB && [ -f "$DB_PATH" ]; then
    rm -f "$DB_PATH"
    ok "Ancienne base supprimée (--rebuild-db)"
fi
if [ ! -f "$DB_PATH" ]; then
    if $CHECK_ONLY; then
        fail "Base absente (lance ./run.sh pour la créer depuis Kaggle)"
    else
        echo "  Création depuis Kaggle (dansbecker/powerlifting-database)..."
        if ! "${PY[@]}" etl.py; then
            rm -f "$DB_PATH"  # pas de base à moitié remplie
            die "Échec de l'ETL (connexion à Kaggle ?)."
        fi
    fi
fi
if [ -f "$DB_PATH" ]; then
    if n="$(db_ok 2>&1)"; then
        ok "$n performances chargées"
    else
        fail "Base invalide : $n. Relance avec --rebuild-db."
    fi
fi

# --- 5. Ports ------------------------------------------------------------------
step "Ports"
START_MLFLOW=true
if port_busy "$MLFLOW_PORT"; then
    if mlflow_up; then
        ok "MLflow déjà lancé sur le port $MLFLOW_PORT : il sera réutilisé"
        START_MLFLOW=false
    else
        fail "Port $MLFLOW_PORT occupé par autre chose que MLflow"
    fi
else
    ok "Port $MLFLOW_PORT libre (MLflow)"
fi
port_busy "$STREAMLIT_PORT" && fail "Port $STREAMLIT_PORT occupé (Streamlit déjà lancé ?)" || ok "Port $STREAMLIT_PORT libre (Streamlit)"
if $WITH_CHAT; then
    port_busy "$GRADIO_PORT" && fail "Port $GRADIO_PORT occupé (chat déjà lancé ?)" || ok "Port $GRADIO_PORT libre (Gradio)"
fi

# --- Bilan ---------------------------------------------------------------------
step "Bilan"
echo "  $ERRORS erreur(s), $WARNINGS avertissement(s)"
[ "$ERRORS" -eq 0 ] || die "Corrige les erreurs ci-dessus avant de lancer le projet."
if $CHECK_ONLY; then
    ok "Tout est prêt. Lance ./run.sh pour démarrer."
    exit 0
fi

# --- 6. Lancement --------------------------------------------------------------
step "Lancement"
mkdir -p "$LOG_DIR"
export MLFLOW_TRACKING_URI="http://localhost:$MLFLOW_PORT"
export MLFLOW_DISABLE_AGENT_HINT=1
export GRADIO_SERVER_PORT="$GRADIO_PORT"

# Chaque service tourne dans son propre groupe de processus pour pouvoir
# tout arrêter proprement (MLflow lance plusieurs workers).
set -m
PIDS=()
cleanup() {
    local rc=$?
    trap - INT TERM EXIT
    echo; echo "Arrêt des services..."
    for pid in "${PIDS[@]}"; do kill -- "-$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
    exit "$rc"
}
trap cleanup INT TERM EXIT

start() {
    local name="$1"; shift
    "$@" >"$LOG_DIR/$name.log" 2>&1 &
    PIDS+=("$!")
}

wait_for() {
    local name="$1" check="$2" pid="$3"
    for _ in $(seq 1 60); do
        eval "$check" && return 0
        kill -0 "$pid" 2>/dev/null || { tail -20 "$LOG_DIR/$name.log" >&2; die "$name s'est arrêté au démarrage (voir $LOG_DIR/$name.log)."; }
        sleep 1
    done
    die "$name ne répond pas après 60 s (voir $LOG_DIR/$name.log)."
}

if $START_MLFLOW; then
    start mlflow uv run --no-sync mlflow server --host 127.0.0.1 --port "$MLFLOW_PORT"
    wait_for mlflow mlflow_up "${PIDS[-1]}"
fi
ok "MLflow démarré (port $MLFLOW_PORT)"

start streamlit uv run --no-sync streamlit run app.py \
    --server.port "$STREAMLIT_PORT" --server.headless true
wait_for streamlit "port_busy $STREAMLIT_PORT" "${PIDS[-1]}"
ok "Streamlit démarré (port $STREAMLIT_PORT)"

if $WITH_CHAT; then
    start gradio uv run --no-sync python groq_chat.py
    wait_for gradio "port_busy $GRADIO_PORT" "${PIDS[-1]}"
    ok "Chat démarré (port $GRADIO_PORT)"
fi

step "Liens"
echo "  Streamlit   http://localhost:$STREAMLIT_PORT"
$WITH_CHAT && echo "  Chat        http://localhost:$GRADIO_PORT"
COMPARISON_URL="$("${PY[@]}" -c 'import ml; print(ml.latest_comparison_url() or "")' 2>/dev/null || true)"
if [ -n "$COMPARISON_URL" ]; then
    echo "  Comparaison des modèles (dernier entraînement enregistré) :"
    echo "  $COMPARISON_URL"
    echo
    echo "  ${YELLOW}!${RESET} Ce lien pointe vers un entraînement précédent. Une prédiction dans Streamlit"
    echo "    (bouton « Lancer une prédiction aléatoire » ou page « Prédiction ML ») réentraîne les"
    echo "    modèles et en enregistre un nouveau : utilise alors le bouton « Comparaison des modèles »"
    echo "    de Streamlit, qui pointe toujours vers le dernier."
else
    echo "  Comparaison des modèles : pas encore disponible."
    echo
    echo "  ${YELLOW}!${RESET} Lance d'abord une prédiction dans Streamlit (bouton « Lancer une prédiction"
    echo "    aléatoire » ou page « Prédiction ML ») : les modèles s'entraînent et s'enregistrent dans"
    echo "    MLflow. D'ici là, il n'y a rien à comparer et le chat ne connaît pas les résultats des"
    echo "    modèles. Le lien apparaîtra ensuite via le bouton « Comparaison des modèles » de Streamlit."
fi

echo
echo "Journaux dans $LOG_DIR/. Ctrl+C pour tout arrêter."
# Si un service s'arrête, on arrête les autres.
wait -n "${PIDS[@]}" || true
echo "${YELLOW}Un service s'est arrêté, voir $LOG_DIR/.${RESET}"
