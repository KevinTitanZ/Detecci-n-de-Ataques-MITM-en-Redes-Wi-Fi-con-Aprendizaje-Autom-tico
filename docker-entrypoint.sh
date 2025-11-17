#!/bin/bash
set -e

# Colores para logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función de logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    MITM DETECTOR v1.0                       ║"
echo "║              Detección en Tiempo Real con IA                ║"
echo "║                   Kevin Ordoñez - ESPE                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Verificar permisos
if [[ $EUID -ne 0 ]] && [[ "$1" == "detector" || "$1" == "capturer" || "$1" == "attacker" ]]; then
    warn "Algunos comandos requieren permisos de root"
fi

# Crear directorios si no existen
log "Creando estructura de directorios..."
mkdir -p /app/{data/{raw,processed},models,results,alerts,logs,config}

# Verificar modelo entrenado
if [[ "$1" == "detector" ]] && [[ ! -f "/app/models/mitm_detector.h5" ]]; then
    error "Modelo no encontrado. Ejecuta primero el entrenamiento:"
    echo "docker-compose --profile training up mitm-trainer"
    exit 1
fi

# Verificar dataset para entrenamiento
if [[ "$1" == "trainer" ]] && [[ ! -f "/app/data/processed/dataset_features.csv" ]]; then
    warn "Dataset no encontrado. Ejecutando extracción de características..."
    python3 /app/scripts/extract_features.py
fi

# Configurar interfaz de red si es necesario
if [[ "$1" == "detector" || "$1" == "capturer" || "$1" == "attacker" ]]; then
    INTERFACE=${INTERFACE:-wlan0}
    
    if command -v iwconfig >/dev/null 2>&1; then
        if iwconfig "$INTERFACE" >/dev/null 2>&1; then
            log "Interfaz $INTERFACE detectada"
        else
            warn "Interfaz $INTERFACE no encontrada. Interfaces disponibles:"
            iwconfig 2>/dev/null | grep -o "^[a-zA-Z0-9]*" || echo "No hay interfaces wireless"
        fi
    fi
fi

# Ejecutar comando según el modo
case "$1" in
    "detector")
        log "🚀 Iniciando detector en tiempo real..."
        shift
        exec python3 /app/scripts/real_time_detector_optimized.py "$@"
        ;;
    
    "trainer")
        log "🧠 Iniciando entrenamiento de modelos..."
        shift
        python3 /app/scripts/train_model.py
        python3 /app/scripts/calibrate_threshold_fixed.py
        log "✅ Entrenamiento completado"
        ;;
    
    "comparator")
        log "📊 Iniciando comparación de modelos..."
        shift
        exec python3 /app/scripts/compare_models.py "$@"
        ;;
    
    "capturer")
        log "📡 Iniciando captura de tráfico..."
        shift
        exec python3 /app/scripts/capture_traffic.py "$@"
        ;;
    
    "attacker")
        log "⚔️ Iniciando simulador de ataques..."
        shift
        exec python3 /app/scripts/attack_simulator.py "$@"
        ;;
    
    "api")
        log "🌐 Iniciando API web..."
        shift
        # Futuro: exec python3 /app/scripts/api_server.py "$@"
        echo "API no implementada aún"
        sleep infinity
        ;;
    
    "bash"|"sh")
        log "🐚 Iniciando shell interactivo..."
        exec /bin/bash
        ;;
    
    *)
        log "Comandos disponibles:"
        echo "  detector   - Detector en tiempo real"
        echo "  trainer    - Entrenar modelos"
        echo "  comparator - Comparar modelos"
        echo "  capturer   - Capturar tráfico"
        echo "  attacker   - Simular ataques"
        echo "  api        - API web (futuro)"
        echo "  bash       - Shell interactivo"
        
        if [[ $# -gt 0 ]]; then
            log "Ejecutando comando personalizado: $*"
            exec "$@"
        else
            exec /bin/bash
        fi
        ;;
esac
