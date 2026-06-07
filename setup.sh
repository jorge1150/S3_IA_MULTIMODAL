#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# setup.sh — Instalación completa S3 IA Multimodal
# Probado en: macOS Intel x86_64 · Python 3.12
#
# Uso: bash setup.sh
# ═══════════════════════════════════════════════════════════════

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║      S3 IA MULTIMODAL — Instalación Automática          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Verificar Python 3.12 ─────────────────────────────────
# torch 2.2.x no tiene wheel para Python 3.13 en macOS Intel.
echo "[1/8] Buscando Python 3.12..."

PYTHON=""
for candidate in python3.12 /usr/local/bin/python3.12 /opt/homebrew/bin/python3.12 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1 | grep -oE "3\.12\.[0-9]+")
        if [ -n "$ver" ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "   ✗ Python 3.12 no encontrado."
    echo "   Instálalo con: brew install python@3.12"
    echo "   O descárgalo desde: https://www.python.org/downloads/release/python-3129/"
    exit 1
fi

echo "   ✓ Usando: $($PYTHON --version)"

# ── 2. Crear entorno virtual ─────────────────────────────────
echo "[2/8] Creando entorno virtual (Python 3.12)..."
if [ -d "venv" ]; then
    VENV_VER=$(venv/bin/python --version 2>&1 | grep -oE "3\.[0-9]+")
    if [[ "$VENV_VER" != "3.12" ]]; then
        echo "   ⚠ venv existente es Python $VENV_VER. Recreando con Python 3.12..."
        rm -rf venv
        $PYTHON -m venv venv
    else
        echo "   ✓ venv Python 3.12 ya existe."
    fi
else
    $PYTHON -m venv venv
    echo "   ✓ venv creado."
fi
source venv/bin/activate

# ── 3. Actualizar pip ────────────────────────────────────────
echo "[3/8] Actualizando pip..."
pip install --upgrade pip -q
echo "   ✓ pip $(pip --version | awk '{print $2}')"

# ── 4. PortAudio (macOS, requerido por sounddevice) ──────────
echo "[4/8] Verificando PortAudio..."
if command -v brew &>/dev/null; then
    if brew list portaudio &>/dev/null 2>&1; then
        echo "   ✓ PortAudio ya instalado."
    else
        echo "   Instalando PortAudio via Homebrew..."
        brew install portaudio
        echo "   ✓ PortAudio instalado."
    fi
else
    echo "   ⚠ Homebrew no encontrado. Instálalo desde https://brew.sh"
    echo "     Luego: brew install portaudio"
fi

# ── 5. Instalar dependencias Python ──────────────────────────
# Orden importante: numpy<2 antes que torch y opencv
echo "[5/8] Instalando dependencias Python..."

echo "   → Núcleo (numpy, scipy, Pillow)..."
pip install "numpy<2" requests scipy sounddevice Pillow python-dotenv -q

echo "   → PyTorch 2.2.2 (puede tardar ~2 min)..."
pip install "torch==2.2.2" "torchvision==0.17.2" -q

echo "   → OpenCLIP, faster-whisper, ChromaDB..."
pip install open_clip_torch faster-whisper "chromadb>=0.5.0" -q

echo "   → OpenCV 4.8 (compatible con numpy<2)..."
pip install "opencv-python==4.8.1.78" -q

echo "   → Piper TTS (>=1.4.0) y Gradio 6.16.0..."
pip install "piper-tts>=1.4.0" "gradio==6.16.0" -q

echo "   ✓ Todas las dependencias instaladas."

# ── 6. Verificar imports ─────────────────────────────────────
echo "[6/8] Verificando imports críticos..."
python -c "
import os, warnings
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch, open_clip, chromadb, faster_whisper, gradio, sounddevice, cv2
from piper import PiperVoice
print(f'   torch={torch.__version__}')
print(f'   gradio={gradio.__version__}')
print(f'   chromadb={chromadb.__version__}')
print('   ✓ Todos los módulos importan correctamente.')
"

# ── 7. Ollama ────────────────────────────────────────────────
echo "[7/8] Verificando Ollama..."
if ! command -v ollama &>/dev/null; then
    echo "   Instalando Ollama..."
    if command -v brew &>/dev/null; then
        brew install ollama
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
fi
echo "   ✓ Ollama: $(ollama --version 2>/dev/null || echo 'instalado')"

if curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo "   ✓ Ollama activo. Descargando modelos..."
    ollama pull tinyllama
    ollama pull moondream
    echo "   ✓ Modelos descargados."
else
    echo ""
    echo "   ⚠ Ollama no está ejecutándose."
    echo "   Antes de usar el sistema, en otra terminal ejecuta:"
    echo "       ollama serve"
    echo "   Luego descarga los modelos:"
    echo "       ollama pull tinyllama"
    echo "       ollama pull moondream"
fi

# ── 8. Piper TTS + Base vectorial ────────────────────────────
echo "[8/8] Descargando modelo TTS y construyendo base vectorial..."
python audio/download_piper.py
python rag/build_db.py

# ── Resumen final ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           ✓  INSTALACIÓN COMPLETADA                      ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Para iniciar el sistema:                                ║"
echo "║    1. source venv/bin/activate                           ║"
echo "║    2. ollama serve    ← en otra terminal                 ║"
echo "║    3. python app.py                                      ║"
echo "║                                                          ║"
echo "║  Interfaz: http://localhost:7864                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
